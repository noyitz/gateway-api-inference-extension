use crate::cycle_state::CycleState;
use crate::error::PluginError;
use crate::inference_message::{InferenceRequest, InferenceResponse};

pub trait RequestProcessor: Send + Sync {
    fn name(&self) -> &str;

    fn process_request(
        &self,
        cycle_state: &mut CycleState,
        request: &mut InferenceRequest,
    ) -> Result<(), PluginError>;
}

pub trait ResponseProcessor: Send + Sync {
    fn name(&self) -> &str;

    fn process_response(
        &self,
        cycle_state: &mut CycleState,
        response: &mut InferenceResponse,
    ) -> Result<(), PluginError>;
}
