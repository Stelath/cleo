use std::sync::Arc;
use std::thread;

use crossbeam_channel::Receiver;
use parking_lot::RwLock;

use crate::error::{Result, VitureError};
use crate::sdk::callbacks::depth_from_packet;
use crate::types::{DepthFrame, StereoCameraPacket};

#[derive(Clone)]
pub struct DepthStream {
    latest: Arc<RwLock<Option<StereoCameraPacket>>>,
}

impl DepthStream {
    pub fn new(rx: Receiver<StereoCameraPacket>) -> Self {
        let latest = Arc::new(RwLock::new(None));
        let latest_for_thread = latest.clone();
        thread::spawn(move || {
            while let Ok(packet) = rx.recv() {
                *latest_for_thread.write() = Some(packet);
            }
        });
        Self { latest }
    }

    pub fn capture(&self, side: &str) -> Result<DepthFrame> {
        let packet = self
            .latest
            .read()
            .clone()
            .ok_or(VitureError::StreamEmpty("Depth stream has no frames yet"))?;
        Ok(depth_from_packet(&packet, side))
    }
}
