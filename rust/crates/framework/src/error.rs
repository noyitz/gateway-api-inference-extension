use std::fmt;

#[derive(Debug, Clone)]
pub enum ErrorCode {
    BadRequest,
    NotFound,
    Internal,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::BadRequest => write!(f, "bad_request"),
            ErrorCode::NotFound => write!(f, "not_found"),
            ErrorCode::Internal => write!(f, "internal"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{code}: {msg}")]
pub struct PluginError {
    pub code: ErrorCode,
    pub msg: String,
}

impl PluginError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::BadRequest,
            msg: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::NotFound,
            msg: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::Internal,
            msg: msg.into(),
        }
    }

    pub fn http_status_code(&self) -> u16 {
        match self.code {
            ErrorCode::BadRequest => 400,
            ErrorCode::NotFound => 404,
            ErrorCode::Internal => 500,
        }
    }
}
