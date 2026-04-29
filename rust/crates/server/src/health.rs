use std::net::SocketAddr;

use tonic::transport::Server;
use tonic_health::server::health_reporter;
use tracing::info;

pub async fn serve_health(port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (mut health_reporter, health_service) = health_reporter();

    health_reporter
        .set_serving::<ext_proc_proto::envoy::service::ext_proc::v3::external_processor_server::ExternalProcessorServer<crate::ext_proc_handler::ExtProcServer>>()
        .await;

    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    info!(port = port, "Starting gRPC health check server");

    Server::builder()
        .add_service(health_service)
        .serve(addr)
        .await?;

    Ok(())
}
