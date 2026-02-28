use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tauri error: {0}")]
    Tauri(#[from] tauri::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error("audio error: {0}")]
    Audio(String),
    #[error("display not found")]
    DisplayNotFound,
    #[error("configuration error: {0}")]
    Config(String),
    #[error("protocol parse error: {0}")]
    ProtocolParse(String),
    #[error("grpc error: {0}")]
    Grpc(String),
}

pub type Result<T> = std::result::Result<T, AppError>;
