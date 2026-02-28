#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Gen1,
    Gen2,
    Carina,
    Unknown(i32),
}

impl From<i32> for DeviceType {
    fn from(value: i32) -> Self {
        match value {
            0 => DeviceType::Gen1,
            1 => DeviceType::Gen2,
            2 => DeviceType::Carina,
            other => DeviceType::Unknown(other),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub firmware_version: String,
    pub product_id: i32,
}

#[derive(Debug, Clone)]
pub struct IMUReading {
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
    pub quaternion: [f32; 4],
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub struct RGBFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    pub timestamp_s: f64,
}

#[derive(Debug, Clone)]
pub struct DepthFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
    pub timestamp_s: f64,
}

#[derive(Debug, Clone)]
pub struct StereoCameraPacket {
    pub left0: Vec<u8>,
    pub right0: Vec<u8>,
    pub left1: Vec<u8>,
    pub right1: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub timestamp_s: f64,
}
