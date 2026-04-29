use std::collections::HashMap;

use futures::StreamExt;
use k8s_openapi::api::core::v1::Secret;
use kube::api::Api;
use kube::runtime::watcher;
use kube::runtime::WatchStreamExt;
use kube::Client;
use tracing::{info, warn};

use super::secret_store::SecretStore;

const MANAGED_LABEL: &str = "inference.networking.k8s.io/bbr-managed";

pub async fn run_secret_watcher(client: Client, store: SecretStore) {
    let api: Api<Secret> = Api::all(client);
    let config = watcher::Config::default().labels(&format!("{}=true", MANAGED_LABEL));

    info!("Starting Secret watcher with label selector {}=true", MANAGED_LABEL);

    let mut stream = watcher::watcher(api, config).applied_objects().boxed();

    loop {
        match stream.next().await {
            Some(Ok(secret)) => {
                let name = secret.metadata.name.clone().unwrap_or_default();
                let namespace = secret.metadata.namespace.clone().unwrap_or_default();
                let key = format!("{}/{}", namespace, name);

                let has_label = secret
                    .metadata
                    .labels
                    .as_ref()
                    .and_then(|l| l.get(MANAGED_LABEL))
                    .map(|v| v == "true")
                    .unwrap_or(false);

                if secret.metadata.deletion_timestamp.is_some() || !has_label {
                    info!(key = %key, "Secret deleted or label removed");
                    store.delete(&key);
                    continue;
                }

                let mut credentials = HashMap::new();
                if let Some(data) = &secret.data {
                    for (field, value) in data {
                        credentials.insert(field.clone(), String::from_utf8_lossy(&value.0).to_string());
                    }
                }

                match store.add_or_update(&key, credentials) {
                    Ok(()) => info!(key = %key, "Updated secret store"),
                    Err(e) => warn!(key = %key, error = %e, "Failed to update secret store"),
                }
            }
            Some(Err(e)) => {
                warn!(error = %e, "Secret watcher error");
            }
            None => {
                warn!("Secret watcher stream ended");
                break;
            }
        }
    }
}
