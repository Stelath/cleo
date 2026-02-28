use std::sync::Arc;

use parking_lot::RwLock;

use crate::types::RGBFrame;

#[derive(Clone, Default)]
pub struct FrameStore {
    latest: Arc<RwLock<Option<RGBFrame>>>,
}

impl FrameStore {
    pub fn set(&self, frame: RGBFrame) {
        *self.latest.write() = Some(frame);
    }

    pub fn get(&self) -> Option<RGBFrame> {
        self.latest.read().clone()
    }
}
