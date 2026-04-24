pub mod ext_proc_handler;

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use ext_proc_proto::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer;
use ipp_framework::plugin::{RequestProcessor, ResponseProcessor};
use ipp_k8s_plugins::apikey_injection::secret_store::SecretStore;
use ipp_k8s_plugins::apikey_injection::ApiKeyInjectionPlugin;
use ipp_k8s_plugins::body_field_to_header::BodyFieldToHeaderPlugin;
use ipp_k8s_plugins::model_provider_resolver::model_store::ModelInfoStore;
use ipp_k8s_plugins::model_provider_resolver::ModelProviderResolverPlugin;
use ipp_translators::api_translation_plugin::{ApiTranslationPlugin, VertexOpenAiConfig};
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

    // --- Build plugin chain ---
    let model_store = ModelInfoStore::new();
    let secret_store = SecretStore::new();

    // Plugin 1: body-field-to-header (extracts model → X-Gateway-Model-Name)
    let body_to_header =
        BodyFieldToHeaderPlugin::new("model", "X-Gateway-Model-Name")?;

    // Plugin 2: model-provider-resolver
    let model_resolver = ModelProviderResolverPlugin::new(model_store.clone());

    // Plugin 3: api-translation
    let vertex_config = match (&cli.vertex_project, &cli.vertex_location, &cli.vertex_endpoint) {
        (Some(project), Some(location), Some(endpoint)) => Some(VertexOpenAiConfig {
            project: project.clone(),
            location: location.clone(),
            endpoint: endpoint.clone(),
        }),
        _ => {
            warn!("Vertex OpenAI config not provided, vertex-openai provider will not be available");
            None
        }
    };
    let api_translation = ApiTranslationPlugin::new(vertex_config)?;

    // Plugin 4: apikey-injection
    let apikey_injection = ApiKeyInjectionPlugin::new(secret_store.clone());

    let request_plugins: Vec<Box<dyn RequestProcessor>> = vec![
        Box::new(body_to_header),
        Box::new(model_resolver),
        Box::new(api_translation),
        Box::new(apikey_injection),
    ];

    // api-translation is both request and response processor
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

    let response_plugins: Vec<Box<dyn ResponseProcessor>> = vec![
        Box::new(api_translation_resp),
    ];

    // --- Start kube-rs reconcilers ---
    let kube_client = match kube::Client::try_default().await {
        Ok(c) => {
            info!("Connected to Kubernetes API");
            Some(c)
        }
        Err(e) => {
            warn!(error = %e, "Failed to connect to Kubernetes API, running without reconcilers");
            None
        }
    };

    if let Some(client) = kube_client {
        let model_store_clone = model_store.clone();
        let client_clone = client.clone();
        tokio::spawn(async move {
            ipp_k8s_plugins::model_provider_resolver::reconciler::run_external_model_watcher(
                client_clone,
                model_store_clone,
            )
            .await;
        });

        let secret_store_clone = secret_store.clone();
        tokio::spawn(async move {
            ipp_k8s_plugins::apikey_injection::reconciler::run_secret_watcher(
                client,
                secret_store_clone,
            )
            .await;
        });

        info!("Started ExternalModel and Secret reconcilers");
    }

    // --- Start gRPC server ---
    let ext_proc = ExtProcServer::new(request_plugins, response_plugins);
    let addr: SocketAddr = format!("0.0.0.0:{}", cli.grpc_port).parse()?;

    info!(port = cli.grpc_port, "Starting gRPC ext_proc server");

    Server::builder()
        .add_service(ExternalProcessorServer::new(ext_proc))
        .serve(addr)
        .await?;

    Ok(())
}
