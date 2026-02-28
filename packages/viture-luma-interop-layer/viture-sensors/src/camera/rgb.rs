use crate::camera::frame::FrameStore;
use crate::error::{Result, VitureError};
use crate::sdk::depth::DepthStream;
use crate::types::RGBFrame;

#[derive(Clone)]
pub struct RGBCamera {
    store: FrameStore,
    depth: DepthStream,
}

impl RGBCamera {
    pub fn new(depth: DepthStream) -> Self {
        Self {
            store: FrameStore::default(),
            depth,
        }
    }

    pub fn capture(&self) -> Result<RGBFrame> {
        if let Some(frame) = self.store.get() {
            return Ok(frame);
        }

        let depth = self.depth.capture("left")?;
        let mut rgb = Vec::with_capacity(depth.width * depth.height * 3);
        for px in depth.data {
            let v = (px & 0xFF) as u8;
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
        let frame = RGBFrame {
            width: depth.width,
            height: depth.height,
            data: rgb,
            timestamp_s: depth.timestamp_s,
        };
        self.store.set(frame.clone());
        Ok(frame)
    }

    pub fn stream_latest(&self) -> Result<RGBFrame> {
        self.store
            .get()
            .ok_or(VitureError::StreamEmpty("RGB stream has no frames yet"))
    }
}
