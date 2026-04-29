pub mod ext_proc_handler;
mod health;
mod metrics;

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use ext_proc_proto::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer;
use ipp_framework::plugin::{RequestProcessor, ResponseProcessor};
use tonic::transport::Server;
use tracing::{error, info, warn};

use ext_proc_handler::ExtProcServer;

#[derive(Parser, Debug)]
#[command(
    name = "ipp-server",
    about = "Rust ext_proc server for AI Gateway payload processing"
)]
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

async fn build_plugins(
    cli: &Cli,
) -> Result<(
    Vec<Box<dyn RequestProcessor>>,
    Vec<Box<dyn ResponseProcessor>>,
)> {
    use ipp_plugins::api_translation::{ApiTranslationPlugin, VertexOpenAiConfig};
    use ipp_plugins::apikey_injection::secret_store::SecretStore;
    use ipp_plugins::apikey_injection::ApiKeyInjectionPlugin;
    use ipp_plugins::body_field_to_header::BodyFieldToHeaderPlugin;

    let secret_store = SecretStore::new();

    let body_to_header = BodyFieldToHeaderPlugin::new("model", "X-Gateway-Model-Name")?;

    let vertex_config = match (
        &cli.vertex_project,
        &cli.vertex_location,
        &cli.vertex_endpoint,
    ) {
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
    let api_translation = ApiTranslationPlugin::new(vertex_config.clone())?;
    let apikey_injection = ApiKeyInjectionPlugin::new(secret_store.clone());

    let request_plugins: Vec<Box<dyn RequestProcessor>> = vec![
        Box::new(body_to_header),
        Box::new(api_translation),
        Box::new(apikey_injection),
    ];

    let api_translation_resp = ApiTranslationPlugin::new(vertex_config)?;
    let response_plugins: Vec<Box<dyn ResponseProcessor>> = vec![Box::new(api_translation_resp)];

    // Start kube-rs reconcilers
    let kube_client = kube::Client::try_default().await.ok();
    if let Some(client) = kube_client {
        let ss = secret_store.clone();
        tokio::spawn(async move {
            ipp_plugins::apikey_injection::reconciler::run_secret_watcher(client, ss).await;
        });
        info!("Started Secret reconciler");
    }

    info!("Using upstream generic plugins");
    Ok((request_plugins, response_plugins))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!("Starting IPP Rust ext_proc server");

    let (request_plugins, response_plugins) = build_plugins(&cli).await?;

    let health_port = cli.health_port;
    tokio::spawn(async move {
        if let Err(e) = health::serve_health(health_port).await {
            error!(error = %e, "Health server failed");
        }
    });

    let metrics_instance = metrics::Metrics::new();
    let metrics_clone = metrics_instance.clone();
    tokio::spawn(async move {
        if let Err(e) = metrics::serve_metrics(9090, metrics_clone).await {
            error!(error = %e, "Metrics server failed");
        }
    });

    let ext_proc = ExtProcServer::new(request_plugins, response_plugins);
    let addr: SocketAddr = format!("0.0.0.0:{}", cli.grpc_port).parse()?;

    info!(
        grpc_port = cli.grpc_port,
        health_port = cli.health_port,
        metrics_port = 9090,
        "Starting gRPC ext_proc server"
    );

    let shutdown = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
        info!("Received shutdown signal, draining connections...");
    };

    Server::builder()
        .add_service(ExternalProcessorServer::new(ext_proc))
        .serve_with_shutdown(addr, shutdown)
        .await?;

    info!("Server shutdown complete");
    Ok(())
}
