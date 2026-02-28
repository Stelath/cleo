use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::errors::{AppError, Result};

pub const DEFAULT_GRPC_ADDRESS: &str = "http://localhost:50055";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HudConfig {
    pub audio_device_id: Option<String>,
}

pub fn runtime_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| AppError::Config("unable to locate home directory".to_string()))?;
    Ok(home.join(".viture-hud"))
}

pub fn ensure_runtime_dir() -> Result<PathBuf> {
    let dir = runtime_dir()?;
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn load_config(path: &Path) -> Result<HudConfig> {
    if !path.exists() {
        return Ok(HudConfig::default());
    }

    let raw = fs::read_to_string(path)?;
    let config = serde_json::from_str::<HudConfig>(&raw)?;
    Ok(config)
}

pub fn save_config(path: &Path, config: &HudConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let serialized = serde_json::to_vec_pretty(config)?;
    fs::write(path, serialized)?;
    Ok(())
}
