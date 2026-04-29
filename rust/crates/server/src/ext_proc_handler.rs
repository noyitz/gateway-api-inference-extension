use std::sync::Arc;
use std::time::Duration;

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
use tracing::{error, warn};

const STREAM_TIMEOUT: Duration = Duration::from_secs(30);

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
        let (tx, rx) = mpsc::channel(4);

        let req_plugins = self.request_plugins.clone();
        let resp_plugins = self.response_plugins.clone();

        tokio::spawn(async move {
            let mut cycle_state = CycleState::new();
            let mut inference_request = InferenceRequest::new();
            let mut inference_response = InferenceResponse::new();
            let mut request_body_buf: Vec<u8> = Vec::new();
            let mut response_body_buf: Vec<u8> = Vec::new();

            loop {
                let msg = match tokio::time::timeout(STREAM_TIMEOUT, stream.message()).await {
                    Ok(Ok(Some(msg))) => msg,
                    Ok(Ok(None)) => break, // stream closed
                    Ok(Err(e)) => {
                        error!(error = %e, "gRPC stream error");
                        break;
                    }
                    Err(_) => {
                        warn!("Stream timeout after {}s", STREAM_TIMEOUT.as_secs());
                        break;
                    }
                };
                let response = match msg.request {
                    Some(processing_request::Request::RequestHeaders(ref headers)) => {
                        extract_headers(headers, &mut inference_request.inner);

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
                            match serde_json::from_slice::<serde_json::Value>(&request_body_buf) {
                                Ok(parsed) => inference_request.inner.body = parsed,
                                Err(e) => warn!(error = %e, size = request_body_buf.len(), "Failed to parse request body as JSON"),
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
                            match serde_json::from_slice::<serde_json::Value>(&response_body_buf) {
                                Ok(parsed) => inference_response.inner.body = parsed,
                                Err(e) => warn!(error = %e, size = response_body_buf.len(), "Failed to parse response body as JSON"),
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

#[inline]
fn extract_headers(headers: &HttpHeaders, msg: &mut InferenceMessage) {
    if let Some(ref header_map) = headers.headers {
        msg.headers.reserve(header_map.headers.len());
        for hv in &header_map.headers {
            if !hv.key.is_empty() {
                let value = if !hv.raw_value.is_empty() {
                    String::from_utf8_lossy(&hv.raw_value).into_owned()
                } else {
                    hv.value.clone()
                };
                msg.headers.insert(hv.key.clone(), value);
            }
        }
    }
}

#[inline]
fn build_mutations(msg: &InferenceMessage) -> (Option<HeaderMutation>, Option<BodyMutation>) {
    let mutated_headers = msg.mutated_headers();
    let removed_headers = msg.removed_headers();

    let header_mutation = if mutated_headers.is_empty() && removed_headers.is_empty() {
        None
    } else {
        Some(HeaderMutation {
            set_headers: mutated_headers
                .iter()
                .map(|(k, v)| HeaderValueOption {
                    header: Some(HeaderValue {
                        key: k.clone(),
                        raw_value: v.as_bytes().to_vec(),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            remove_headers: removed_headers,
            ..Default::default()
        })
    };

    let body_mutation = if !msg.body_mutated() {
        None
    } else {
        serde_json::to_vec(&msg.body).ok().map(|bytes| BodyMutation {
            mutation: Some(body_mutation::Mutation::Body(bytes)),
        })
    };

    (header_mutation, body_mutation)
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

    // Serialize body once, use for both content-length and body mutation
    if request.body_mutated() {
        if let Ok(body_bytes) = serde_json::to_vec(&request.body) {
            request.set_header("content-length", body_bytes.len().to_string());
            // Store pre-serialized body to avoid double serialization
            let common = CommonResponse {
                header_mutation: {
                    let mutated = request.mutated_headers();
                    let removed = request.removed_headers();
                    if mutated.is_empty() && removed.is_empty() {
                        None
                    } else {
                        Some(HeaderMutation {
                            set_headers: mutated.iter().map(|(k, v)| HeaderValueOption {
                                header: Some(HeaderValue {
                                    key: k.clone(),
                                    raw_value: v.as_bytes().to_vec(),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }).collect(),
                            remove_headers: removed,
                            ..Default::default()
                        })
                    }
                },
                body_mutation: Some(BodyMutation {
                    mutation: Some(body_mutation::Mutation::Body(body_bytes)),
                }),
                ..Default::default()
            };
            return Ok(ProcessingResponse {
                response: Some(Response::RequestBody(BodyResponse {
                    response: Some(common),
                })),
                ..Default::default()
            });
        }
    }

    let (header_mutation, body_mutation) = build_mutations(&request.inner);
    Ok(ProcessingResponse {
        response: Some(Response::RequestBody(BodyResponse {
            response: Some(CommonResponse {
                header_mutation,
                body_mutation,
                ..Default::default()
            }),
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

    if response.body_mutated() {
        if let Ok(body_bytes) = serde_json::to_vec(&response.body) {
            response.set_header("content-length", body_bytes.len().to_string());
            let common = CommonResponse {
                header_mutation: {
                    let mutated = response.mutated_headers();
                    let removed = response.removed_headers();
                    if mutated.is_empty() && removed.is_empty() {
                        None
                    } else {
                        Some(HeaderMutation {
                            set_headers: mutated.iter().map(|(k, v)| HeaderValueOption {
                                header: Some(HeaderValue {
                                    key: k.clone(),
                                    raw_value: v.as_bytes().to_vec(),
                                    ..Default::default()
                                }),
                                ..Default::default()
                            }).collect(),
                            remove_headers: removed,
                            ..Default::default()
                        })
                    }
                },
                body_mutation: Some(BodyMutation {
                    mutation: Some(body_mutation::Mutation::Body(body_bytes)),
                }),
                ..Default::default()
            };
            return Ok(ProcessingResponse {
                response: Some(Response::ResponseBody(BodyResponse {
                    response: Some(common),
                })),
                ..Default::default()
            });
        }
    }

    let (header_mutation, body_mutation) = build_mutations(&response.inner);
    Ok(ProcessingResponse {
        response: Some(Response::ResponseBody(BodyResponse {
            response: Some(CommonResponse {
                header_mutation,
                body_mutation,
                ..Default::default()
            }),
        })),
        ..Default::default()
    })
}
