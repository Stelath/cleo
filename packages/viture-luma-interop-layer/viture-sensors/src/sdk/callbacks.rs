use std::slice;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use crossbeam_channel::Sender;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::types::{DepthFrame, IMUReading, StereoCameraPacket};

static IMU_SENDER: Lazy<Mutex<Option<Sender<IMUReading>>>> = Lazy::new(|| Mutex::new(None));
static STEREO_SENDER: Lazy<Mutex<Option<Sender<StereoCameraPacket>>>> =
    Lazy::new(|| Mutex::new(None));
static STATE_SENDER: Lazy<Mutex<Option<Sender<(i32, i32)>>>> = Lazy::new(|| Mutex::new(None));
static CAMERA_CALLBACK_CALLS: AtomicU64 = AtomicU64::new(0);
static CAMERA_CALLBACK_NULL_PTR: AtomicU64 = AtomicU64::new(0);
static CAMERA_CALLBACK_BAD_DIM: AtomicU64 = AtomicU64::new(0);
static CAMERA_CALLBACK_USABLE: AtomicU64 = AtomicU64::new(0);
static CAMERA_LAST_WIDTH: AtomicI32 = AtomicI32::new(0);
static CAMERA_LAST_HEIGHT: AtomicI32 = AtomicI32::new(0);

pub fn install_imu_sender(sender: Sender<IMUReading>) {
    *IMU_SENDER.lock() = Some(sender);
}

pub fn install_stereo_sender(sender: Sender<StereoCameraPacket>) {
    *STEREO_SENDER.lock() = Some(sender);
}

pub fn install_state_sender(sender: Sender<(i32, i32)>) {
    *STATE_SENDER.lock() = Some(sender);
}

pub fn clear_all() {
    *IMU_SENDER.lock() = None;
    *STEREO_SENDER.lock() = None;
    *STATE_SENDER.lock() = None;
}

pub fn reset_camera_debug_counters() {
    CAMERA_CALLBACK_CALLS.store(0, Ordering::Relaxed);
    CAMERA_CALLBACK_NULL_PTR.store(0, Ordering::Relaxed);
    CAMERA_CALLBACK_BAD_DIM.store(0, Ordering::Relaxed);
    CAMERA_CALLBACK_USABLE.store(0, Ordering::Relaxed);
    CAMERA_LAST_WIDTH.store(0, Ordering::Relaxed);
    CAMERA_LAST_HEIGHT.store(0, Ordering::Relaxed);
}

pub fn camera_debug_counters() -> (u64, u64, u64, u64, i32, i32) {
    (
        CAMERA_CALLBACK_CALLS.load(Ordering::Relaxed),
        CAMERA_CALLBACK_NULL_PTR.load(Ordering::Relaxed),
        CAMERA_CALLBACK_BAD_DIM.load(Ordering::Relaxed),
        CAMERA_CALLBACK_USABLE.load(Ordering::Relaxed),
        CAMERA_LAST_WIDTH.load(Ordering::Relaxed),
        CAMERA_LAST_HEIGHT.load(Ordering::Relaxed),
    )
}

unsafe extern "C" fn state_callback(glass_state_id: i32, glass_value: i32) {
    if let Some(tx) = STATE_SENDER.lock().as_ref() {
        let _ = tx.try_send((glass_state_id, glass_value));
    }
}

unsafe extern "C" fn imu_pose_callback(data: *mut f32, timestamp: u64) {
    if data.is_null() {
        return;
    }
    let values = unsafe { slice::from_raw_parts(data as *const f32, 7) };
    let reading = IMUReading {
        accel: [0.0, 0.0, 0.0],
        gyro: [0.0, 0.0, 0.0],
        quaternion: [values[3], values[4], values[5], values[6]],
        timestamp_ns: timestamp,
    };
    if let Some(tx) = IMU_SENDER.lock().as_ref() {
        let _ = tx.try_send(reading);
    }
}

unsafe extern "C" fn imu_raw_callback(data: *mut f32, timestamp: u64, _vsync: u64) {
    if data.is_null() {
        return;
    }
    let values = unsafe { slice::from_raw_parts(data as *const f32, 10) };
    let reading = IMUReading {
        accel: [values[3], values[4], values[5]],
        gyro: [values[0], values[1], values[2]],
        quaternion: [1.0, 0.0, 0.0, 0.0],
        timestamp_ns: timestamp,
    };
    if let Some(tx) = IMU_SENDER.lock().as_ref() {
        let _ = tx.try_send(reading);
    }
}

unsafe extern "C" fn carina_imu_callback(data: *mut f32, timestamp: f64) {
    if data.is_null() {
        return;
    }
    let values = unsafe { slice::from_raw_parts(data as *const f32, 6) };
    let reading = IMUReading {
        accel: [values[0], values[1], values[2]],
        gyro: [values[3], values[4], values[5]],
        quaternion: [1.0, 0.0, 0.0, 0.0],
        timestamp_ns: (timestamp * 1_000_000_000.0) as u64,
    };
    if let Some(tx) = IMU_SENDER.lock().as_ref() {
        let _ = tx.try_send(reading);
    }
}

unsafe extern "C" fn carina_pose_callback(_pose: *mut f32, _timestamp: f64) {}

unsafe extern "C" fn carina_vsync_callback(_timestamp: f64) {}

unsafe extern "C" fn carina_camera_callback(
    image_left0: *mut i8,
    image_right0: *mut i8,
    image_left1: *mut i8,
    image_right1: *mut i8,
    timestamp: f64,
    width: i32,
    height: i32,
) {
    CAMERA_CALLBACK_CALLS.fetch_add(1, Ordering::Relaxed);
    CAMERA_LAST_WIDTH.store(width, Ordering::Relaxed);
    CAMERA_LAST_HEIGHT.store(height, Ordering::Relaxed);

    if width <= 0 || height <= 0 {
        CAMERA_CALLBACK_BAD_DIM.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let px = (width as usize) * (height as usize);

    let left0 = if !image_left0.is_null() {
        unsafe { slice::from_raw_parts(image_left0 as *const u8, px) }.to_vec()
    } else if !image_left1.is_null() {
        unsafe { slice::from_raw_parts(image_left1 as *const u8, px) }.to_vec()
    } else {
        CAMERA_CALLBACK_NULL_PTR.fetch_add(1, Ordering::Relaxed);
        return;
    };

    let right0 = if !image_right0.is_null() {
        unsafe { slice::from_raw_parts(image_right0 as *const u8, px) }.to_vec()
    } else if !image_right1.is_null() {
        unsafe { slice::from_raw_parts(image_right1 as *const u8, px) }.to_vec()
    } else {
        left0.clone()
    };

    let left1 = if !image_left1.is_null() {
        unsafe { slice::from_raw_parts(image_left1 as *const u8, px) }.to_vec()
    } else {
        left0.clone()
    };

    let right1 = if !image_right1.is_null() {
        unsafe { slice::from_raw_parts(image_right1 as *const u8, px) }.to_vec()
    } else {
        right0.clone()
    };

    CAMERA_CALLBACK_USABLE.fetch_add(1, Ordering::Relaxed);
    let packet = StereoCameraPacket {
        left0,
        right0,
        left1,
        right1,
        width: width as usize,
        height: height as usize,
        timestamp_s: timestamp,
    };
    if let Some(tx) = STEREO_SENDER.lock().as_ref() {
        let _ = tx.try_send(packet);
    }
}

pub fn depth_from_packet(packet: &StereoCameraPacket, side: &str) -> DepthFrame {
    let src = match side {
        "right" => &packet.right0,
        _ => &packet.left0,
    };
    let data = src.iter().map(|v| *v as u16).collect();
    DepthFrame {
        width: packet.width,
        height: packet.height,
        data,
        timestamp_s: packet.timestamp_s,
    }
}

pub fn state_callback_ptr() -> unsafe extern "C" fn(i32, i32) {
    state_callback
}

pub fn imu_pose_callback_ptr() -> unsafe extern "C" fn(*mut f32, u64) {
    imu_pose_callback
}

pub fn imu_raw_callback_ptr() -> unsafe extern "C" fn(*mut f32, u64, u64) {
    imu_raw_callback
}

pub fn carina_imu_callback_ptr() -> unsafe extern "C" fn(*mut f32, f64) {
    carina_imu_callback
}

pub fn carina_pose_callback_ptr() -> unsafe extern "C" fn(*mut f32, f64) {
    carina_pose_callback
}

pub fn carina_vsync_callback_ptr() -> unsafe extern "C" fn(f64) {
    carina_vsync_callback
}

pub fn carina_camera_callback_ptr(
) -> unsafe extern "C" fn(*mut i8, *mut i8, *mut i8, *mut i8, f64, i32, i32) {
    carina_camera_callback
}
