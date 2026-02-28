use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use tokio::sync::broadcast;

use crate::audio::CancelToken;
use crate::config::{save_config, HudConfig};
use crate::errors::Result;
use crate::protocol::Event;

#[derive(Clone)]
pub struct AppState {
    event_tx: broadcast::Sender<Event>,
    config_path: Arc<PathBuf>,
    audio_device_id: Arc<RwLock<Option<String>>>,
    audio_cancel: CancelToken,
}

impl AppState {
    pub fn new(config_path: PathBuf, initial_audio_device: Option<String>) -> Self {
        let (event_tx, _) = broadcast::channel(128);

        Self {
            event_tx,
            config_path: Arc::new(config_path),
            audio_device_id: Arc::new(RwLock::new(initial_audio_device)),
            audio_cancel: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn event_tx(&self) -> broadcast::Sender<Event> {
        self.event_tx.clone()
    }

    pub fn selected_audio_device(&self) -> Option<String> {
        self.audio_device_id
            .read()
            .expect("audio lock poisoned")
            .clone()
    }

    pub fn set_selected_audio_device(&self, device_id: Option<String>) -> Result<()> {
        *self.audio_device_id.write().expect("audio lock poisoned") = device_id.clone();
        let config = HudConfig {
            audio_device_id: device_id,
        };
        save_config(self.config_path.as_ref(), &config)
    }

    pub fn cancel_audio(&self) {
        self.audio_cancel.fetch_add(1, Ordering::Relaxed);
    }

    pub fn audio_cancel_token(&self) -> CancelToken {
        self.audio_cancel.clone()
    }
}
