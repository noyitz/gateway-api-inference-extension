use std::sync::Arc;

use ext_proc_proto::envoy::config::core::v3::{HeaderValue, HeaderValueOption};
use ext_proc_proto::envoy::service::ext_proc::v3::processing_response::Response;
use ext_proc_proto::envoy::service::ext_proc::v3::{
    body_mutation, external_processor_server::ExternalProcessor, processing_request,
    BodyMutation, BodyResponse, CommonResponse, HeaderMutation, HeadersResponse,
    HttpHeaders, ProcessingRequest, ProcessingResponse,
};
use ipp_framework::cycle_state::CycleState;
use ipp_framework::inference_message::{InferenceMessage, InferenceRequest, InferenceResponse};
use ipp_framework::plugin::{RequestProcessor, ResponseProcessor};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Status, Streaming};
use tracing::{debug, error, warn};

pub struct ExtProcServer {
    request_plugins: Arc<Vec<Box<dyn RequestProcessor>>>,
    response_plugins: Arc<Vec<Box<dyn ResponseProcessor>>>,
}

impl ExtProcServer {
    pub fn new(
        request_plugins: Vec<Box<dyn RequestProcessor>>,
        response_plugins: Vec<Box<dyn ResponseProcessor>>,
    ) -> Self {
        Self {
            request_plugins: Arc::new(request_plugins),
            response_plugins: Arc::new(response_plugins),
        }
    }
}

#[tonic::async_trait]
impl ExternalProcessor for ExtProcServer {
    type ProcessStream = ReceiverStream<Result<ProcessingResponse, Status>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<tonic::Response<Self::ProcessStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = mpsc::channel(32);

        let req_plugins = self.request_plugins.clone();
        let resp_plugins = self.response_plugins.clone();

        tokio::spawn(async move {
            let mut cycle_state = CycleState::new();
            let mut inference_request = InferenceRequest::new();
            let mut inference_response = InferenceResponse::new();
            let mut request_body_buf: Vec<u8> = Vec::new();
            let mut response_body_buf: Vec<u8> = Vec::new();

            while let Ok(Some(msg)) = stream.message().await {
                let response = match msg.request {
                    Some(processing_request::Request::RequestHeaders(ref headers)) => {
                        extract_headers(headers, &mut inference_request.inner);
                        debug!(path = ?inference_request.headers.get(":path"), "Received request headers");

                        if headers.end_of_stream {
                            run_request_plugins_and_respond(
                                &req_plugins,
                                &mut cycle_state,
                                &mut inference_request,
                            )
                        } else {
                            Ok(ProcessingResponse {
                                response: Some(Response::RequestHeaders(
                                    HeadersResponse::default(),
                                )),
                                ..Default::default()
                            })
                        }
                    }
                    Some(processing_request::Request::RequestBody(ref body)) => {
                        request_body_buf.extend_from_slice(&body.body);

                        if body.end_of_stream {
                            if let Ok(parsed) =
                                serde_json::from_slice::<serde_json::Value>(&request_body_buf)
                            {
                                inference_request.inner.body = parsed;
                            } else {
                                warn!("Failed to parse request body as JSON");
                            }
                            run_request_plugins_and_respond(
                                &req_plugins,
                                &mut cycle_state,
                                &mut inference_request,
                            )
                        } else {
                            continue;
                        }
                    }
                    Some(processing_request::Request::ResponseHeaders(ref headers)) => {
                        extract_headers(headers, &mut inference_response.inner);
                        debug!("Received response headers");

                        if headers.end_of_stream {
                            run_response_plugins_and_respond(
                                &resp_plugins,
                                &mut cycle_state,
                                &mut inference_response,
                            )
                        } else {
                            Ok(ProcessingResponse {
                                response: Some(Response::ResponseHeaders(
                                    HeadersResponse::default(),
                                )),
                                ..Default::default()
                            })
                        }
                    }
                    Some(processing_request::Request::ResponseBody(ref body)) => {
                        response_body_buf.extend_from_slice(&body.body);

                        if body.end_of_stream {
                            if let Ok(parsed) =
                                serde_json::from_slice::<serde_json::Value>(&response_body_buf)
                            {
                                inference_response.inner.body = parsed;
                            } else {
                                debug!("Failed to parse response body as JSON");
                            }
                            run_response_plugins_and_respond(
                                &resp_plugins,
                                &mut cycle_state,
                                &mut inference_response,
                            )
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                match response {
                    Ok(resp) => {
                        if tx.send(Ok(resp)).await.is_err() {
                            break;
                        }
                    }
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        break;
                    }
                }
            }
        });

        Ok(tonic::Response::new(ReceiverStream::new(rx)))
    }
}

fn extract_headers(headers: &HttpHeaders, msg: &mut InferenceMessage) {
    if let Some(ref header_map) = headers.headers {
        for hv in &header_map.headers {
            if !hv.key.is_empty() {
                msg.headers.insert(hv.key.clone(), hv.value.clone());
            }
        }
    }
}

fn build_header_mutation(msg: &impl HasMutations) -> Option<HeaderMutation> {
    let mutated = msg.mutated_headers();
    let removed = msg.removed_headers();
    if mutated.is_empty() && removed.is_empty() {
        return None;
    }

    Some(HeaderMutation {
        set_headers: mutated
            .iter()
            .map(|(k, v)| HeaderValueOption {
                header: Some(HeaderValue {
                    key: k.clone(),
                    value: v.clone(),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect(),
        remove_headers: removed,
        ..Default::default()
    })
}

fn build_body_mutation(msg: &InferenceMessage) -> Option<BodyMutation> {
    if !msg.body_mutated() {
        return None;
    }
    let body_bytes = serde_json::to_vec(&msg.body).ok()?;
    Some(BodyMutation {
        mutation: Some(body_mutation::Mutation::Body(body_bytes)),
    })
}

trait HasMutations {
    fn mutated_headers(&self) -> &std::collections::HashMap<String, String>;
    fn removed_headers(&self) -> Vec<String>;
}

impl HasMutations for InferenceMessage {
    fn mutated_headers(&self) -> &std::collections::HashMap<String, String> {
        self.mutated_headers()
    }
    fn removed_headers(&self) -> Vec<String> {
        self.removed_headers()
    }
}

fn run_request_plugins_and_respond(
    plugins: &[Box<dyn RequestProcessor>],
    cycle_state: &mut CycleState,
    request: &mut InferenceRequest,
) -> Result<ProcessingResponse, Status> {
    for plugin in plugins {
        if let Err(e) = plugin.process_request(cycle_state, request) {
            error!(plugin = plugin.name(), error = %e, "Request plugin failed");
            return Err(Status::internal(e.to_string()));
        }
    }

    let common = CommonResponse {
        header_mutation: build_header_mutation(&request.inner),
        body_mutation: build_body_mutation(&request.inner),
        ..Default::default()
    };

    Ok(ProcessingResponse {
        response: Some(Response::RequestBody(BodyResponse {
            response: Some(common),
        })),
        ..Default::default()
    })
}

fn run_response_plugins_and_respond(
    plugins: &[Box<dyn ResponseProcessor>],
    cycle_state: &mut CycleState,
    response: &mut InferenceResponse,
) -> Result<ProcessingResponse, Status> {
    for plugin in plugins {
        if let Err(e) = plugin.process_response(cycle_state, response) {
            error!(plugin = plugin.name(), error = %e, "Response plugin failed");
            return Err(Status::internal(e.to_string()));
        }
    }

    let common = CommonResponse {
        header_mutation: build_header_mutation(&response.inner),
        body_mutation: build_body_mutation(&response.inner),
        ..Default::default()
    };

    Ok(ProcessingResponse {
        response: Some(Response::ResponseBody(BodyResponse {
            response: Some(common),
        })),
        ..Default::default()
    })
}
