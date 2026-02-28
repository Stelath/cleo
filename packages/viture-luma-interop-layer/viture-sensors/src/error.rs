use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, VitureError>;

#[derive(Debug, Error)]
pub enum VitureError {
    #[error("SDK call failed: {function} returned {code}")]
    SdkCallFailed { function: &'static str, code: i32 },
    #[error("device not connected")]
    NotConnected,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("stream has no data yet: {0}")]
    StreamEmpty(&'static str),
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
    #[error("audio error: {0}")]
    Audio(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<VitureError> for PyErr {
    fn from(value: VitureError) -> Self {
        match value {
            VitureError::InvalidArgument(msg) => PyValueError::new_err(msg),
            VitureError::NotConnected => PyConnectionError::new_err("device is not connected"),
            VitureError::StreamEmpty(msg) => PyRuntimeError::new_err(msg),
            VitureError::SdkCallFailed { function, code } => {
                PyRuntimeError::new_err(format!("{function} failed with code {code}"))
            }
            VitureError::Unsupported(msg) => PyRuntimeError::new_err(msg),
            VitureError::Audio(msg) => PyRuntimeError::new_err(msg),
            VitureError::Internal(msg) => PyRuntimeError::new_err(msg),
        }
    }
}
