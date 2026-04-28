pub mod ext_proc_handler;

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use ext_proc_proto::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer;
use ipp_framework::plugin::{RequestProcessor, ResponseProcessor};
use tonic::transport::Server;
use tracing::{info, warn};

use ext_proc_handler::ExtProcServer;

#[derive(Parser, Debug)]
#[command(name = "ipp-server", about = "Rust ext_proc server for AI Gateway payload processing")]
struct Cli {
    #[arg(long, default_value = "9004", help = "gRPC ext_proc port")]
    grpc_port: u16,

    #[arg(long, default_value = "9005", help = "Health check port")]
    health_port: u16,

    #[arg(long, help = "Vertex OpenAI project")]
    vertex_project: Option<String>,

    #[arg(long, help = "Vertex OpenAI location")]
    vertex_location: Option<String>,

    #[arg(long, help = "Vertex OpenAI endpoint")]
    vertex_endpoint: Option<String>,
}

#[cfg(feature = "go-plugins")]
async fn build_plugins(
    cli: &Cli,
) -> Result<(Vec<Box<dyn RequestProcessor>>, Vec<Box<dyn ResponseProcessor>>)> {
    use ipp_framework::cycle_state::CycleState;
    use ipp_framework::inference_message::{InferenceRequest, InferenceResponse};

    let vertex_config = match (&cli.vertex_project, &cli.vertex_location, &cli.vertex_endpoint) {
        (Some(p), Some(l), Some(e)) => {
            Some(format!(
                r#"{{"project":"{}","location":"{}","endpoint":"{}"}}"#,
                p, l, e
            ))
        }
        _ => None,
    };

    let bridge = ipp_go_ffi::GoPluginBridge::new(vertex_config.as_deref())
        .map_err(|e| anyhow::anyhow!("Go plugin init failed: {}", e))?;

    info!("Using Go plugins via FFI (Config D — Reverse Hybrid)");

    // Leak the bridge into a &'static reference so it can be shared
    let bridge: &'static ipp_go_ffi::GoPluginBridge = Box::leak(Box::new(bridge));

    struct ReqBridge(&'static ipp_go_ffi::GoPluginBridge);
    unsafe impl Send for ReqBridge {}
    unsafe impl Sync for ReqBridge {}
    impl RequestProcessor for ReqBridge {
        fn name(&self) -> &str { "go-plugins-bridge" }
        fn process_request(&self, cs: &mut CycleState, req: &mut InferenceRequest) -> std::result::Result<(), ipp_framework::error::PluginError> {
            self.0.process_request(cs, req)
        }
    }

    struct RespBridge(&'static ipp_go_ffi::GoPluginBridge);
    unsafe impl Send for RespBridge {}
    unsafe impl Sync for RespBridge {}
    impl ResponseProcessor for RespBridge {
        fn name(&self) -> &str { "go-plugins-bridge" }
        fn process_response(&self, cs: &mut CycleState, resp: &mut InferenceResponse) -> std::result::Result<(), ipp_framework::error::PluginError> {
            self.0.process_response(cs, resp)
        }
    }

    Ok((
        vec![Box::new(ReqBridge(bridge))],
        vec![Box::new(RespBridge(bridge))],
    ))
}

#[cfg(not(feature = "go-plugins"))]
async fn build_plugins(
    cli: &Cli,
) -> Result<(Vec<Box<dyn RequestProcessor>>, Vec<Box<dyn ResponseProcessor>>)> {
    use ipp_k8s_plugins::apikey_injection::secret_store::SecretStore;
    use ipp_k8s_plugins::apikey_injection::ApiKeyInjectionPlugin;
    use ipp_k8s_plugins::body_field_to_header::BodyFieldToHeaderPlugin;
    use ipp_k8s_plugins::model_provider_resolver::model_store::ModelInfoStore;
    use ipp_k8s_plugins::model_provider_resolver::ModelProviderResolverPlugin;
    use ipp_translators::api_translation_plugin::{ApiTranslationPlugin, VertexOpenAiConfig};

    let model_store = ModelInfoStore::new();
    let secret_store = SecretStore::new();

    let body_to_header = BodyFieldToHeaderPlugin::new("model", "X-Gateway-Model-Name")?;
    let model_resolver = ModelProviderResolverPlugin::new(model_store.clone());

    let vertex_config = match (&cli.vertex_project, &cli.vertex_location, &cli.vertex_endpoint) {
        (Some(project), Some(location), Some(endpoint)) => Some(VertexOpenAiConfig {
            project: project.clone(),
            location: location.clone(),
            endpoint: endpoint.clone(),
        }),
        _ => {
            warn!("Vertex OpenAI config not provided");
            None
        }
    };
    let api_translation = ApiTranslationPlugin::new(vertex_config)?;
    let apikey_injection = ApiKeyInjectionPlugin::new(secret_store.clone());

    let request_plugins: Vec<Box<dyn RequestProcessor>> = vec![
        Box::new(body_to_header),
        Box::new(model_resolver),
        Box::new(api_translation),
        Box::new(apikey_injection),
    ];

    let api_translation_resp = ApiTranslationPlugin::new(
        match (&cli.vertex_project, &cli.vertex_location, &cli.vertex_endpoint) {
            (Some(p), Some(l), Some(e)) => Some(VertexOpenAiConfig {
                project: p.clone(),
                location: l.clone(),
                endpoint: e.clone(),
            }),
            _ => None,
        },
    )?;

    let response_plugins: Vec<Box<dyn ResponseProcessor>> = vec![Box::new(api_translation_resp)];

    // Start kube-rs reconcilers — kube::Client::try_default is called from main() which is async
    let kube_client = kube::Client::try_default().await.ok();

    if let Some(client) = kube_client {
        let ms = model_store.clone();
        let c = client.clone();
        tokio::spawn(async move {
            ipp_k8s_plugins::model_provider_resolver::reconciler::run_external_model_watcher(c, ms)
                .await;
        });
        let ss = secret_store.clone();
        tokio::spawn(async move {
            ipp_k8s_plugins::apikey_injection::reconciler::run_secret_watcher(client, ss).await;
        });
        info!("Started ExternalModel and Secret reconcilers (Rust)");
    }

    info!("Using Rust plugins (Config C — Full Rust)");
    Ok((request_plugins, response_plugins))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!("Starting IPP Rust ext_proc server");

    let (request_plugins, response_plugins) = build_plugins(&cli).await?;

    let ext_proc = ExtProcServer::new(request_plugins, response_plugins);
    let addr: SocketAddr = format!("0.0.0.0:{}", cli.grpc_port).parse()?;

    info!(port = cli.grpc_port, "Starting gRPC ext_proc server");

    Server::builder()
        .add_service(ExternalProcessorServer::new(ext_proc))
        .serve(addr)
        .await?;

    Ok(())
}
