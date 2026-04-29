use std::net::SocketAddr;

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use tokio::io::AsyncWriteExt;
use tracing::info;

#[derive(Clone)]
pub struct Metrics {
    pub request_duration: HistogramVec,
    pub request_total: IntCounterVec,
    pub plugin_errors: IntCounterVec,
    registry: Registry,
}

impl Metrics {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "ipp_request_duration_seconds",
                "Request processing duration",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["phase"],
        )?;

        let request_total = IntCounterVec::new(
            Opts::new("ipp_requests_total", "Total requests processed"),
            &["status"],
        )?;

        let plugin_errors = IntCounterVec::new(
            Opts::new("ipp_plugin_errors_total", "Plugin errors"),
            &["plugin"],
        )?;

        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(request_total.clone()))?;
        registry.register(Box::new(plugin_errors.clone()))?;

        Ok(Self {
            request_duration,
            request_total,
            plugin_errors,
            registry,
        })
    }

    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        let _ = encoder.encode(&metric_families, &mut buffer);
        String::from_utf8_lossy(&buffer).into_owned()
    }
}

pub async fn serve_metrics(
    port: u16,
    metrics: Metrics,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
    info!(port = port, "Starting metrics server");

    let listener = tokio::net::TcpListener::bind(addr).await?;

    loop {
        let (mut stream, _) = listener.accept().await?;
        let body = metrics.encode();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes()).await;
    }
}
