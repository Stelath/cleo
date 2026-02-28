use std::ffi::{c_void, CString};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{unbounded, Receiver};

use crate::error::{Result, VitureError};
use crate::sdk::callbacks;
use crate::sdk::depth::DepthStream;
use crate::sdk::ffi;
use crate::sdk::imu::IMUStream;
use crate::sdk::safe::{check_non_negative, check_status};
use crate::types::{DeviceInfo, DeviceType, IMUReading, StereoCameraPacket};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Created,
    Initialized,
    Started,
    Stopped,
    Shutdown,
}

pub struct VitureDevice {
    handle: NonNull<c_void>,
    product_id: i32,
    state: parking_lot::Mutex<DeviceState>,
    connected: AtomicBool,
    device_type: parking_lot::RwLock<DeviceType>,
    imu_stream: IMUStream,
    depth_stream: DepthStream,
    _state_rx: Receiver<(i32, i32)>,
}

unsafe impl Send for VitureDevice {}

impl VitureDevice {
    fn sdk_log_level_from_env() -> i32 {
        std::env::var("VITURE_SDK_LOG_LEVEL")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|v| (0..=3).contains(v))
            .unwrap_or(1)
    }

    fn build_initialize_args() -> (Option<CString>, Option<CString>) {
        let custom = std::env::var("VITURE_CUSTOM_CONFIG")
            .ok()
            .and_then(|v| CString::new(v).ok());

        let cache_path = std::env::var("VITURE_CACHE_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("viture-sensors-cache"));
        let _ = std::fs::create_dir_all(&cache_path);
        let cache = CString::new(cache_path.to_string_lossy().to_string()).ok();

        (custom, cache)
    }

    fn product_id_from_env() -> Option<i32> {
        std::env::var("VITURE_PRODUCT_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
    }

    #[cfg(target_os = "macos")]
    fn discover_connected_product_id() -> Option<i32> {
        use std::process::Command;

        let output = Command::new("ioreg")
            .args(["-p", "IOUSB", "-l", "-w", "0"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mut current_is_viture_node = false;
        let mut current_vendor_is_viture = false;

        for line in text.lines() {
            let trimmed = line.trim();

            if trimmed.contains("+-o ") || trimmed.contains("| +-o ") {
                current_is_viture_node = trimmed.to_ascii_lowercase().contains("viture");
                current_vendor_is_viture = false;
            }

            if trimmed.contains("\"USB Vendor Name\"")
                && trimmed.to_ascii_lowercase().contains("viture")
            {
                current_vendor_is_viture = true;
            }

            if trimmed.contains("\"idVendor\"")
                && Self::parse_ioreg_numeric_value(trimmed).is_some_and(|v| v == 13_770)
            {
                current_vendor_is_viture = true;
            }

            if trimmed.contains("\"idProduct\"")
                && (current_is_viture_node || current_vendor_is_viture)
            {
                if let Some(pid) = Self::parse_ioreg_numeric_value(trimmed) {
                    return Some(pid);
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "macos"))]
    fn discover_connected_product_id() -> Option<i32> {
        None
    }

    fn parse_ioreg_numeric_value(line: &str) -> Option<i32> {
        let value = line.split('=').nth(1)?.trim();
        value.parse::<i32>().ok()
    }

    pub fn discover_product_id() -> Option<i32> {
        for product_id in 0..=0xFFFF {
            let valid = unsafe { ffi::xr_device_provider_is_product_id_valid(product_id) };
            if valid {
                return Some(product_id);
            }
        }
        None
    }

    pub fn new(product_id: Option<i32>) -> Result<Self> {
        // Default to error-only SDK logs to reduce noisy info messages.
        unsafe {
            ffi::xr_device_provider_set_log_level(Self::sdk_log_level_from_env());
        }

        let chosen_product_id = product_id
            .or_else(Self::product_id_from_env)
            .or_else(Self::discover_connected_product_id)
            .or_else(Self::discover_product_id)
            .ok_or_else(|| {
                VitureError::InvalidArgument("No valid VITURE product id found".to_string())
            })?;

        let handle = unsafe { ffi::xr_device_provider_create(chosen_product_id) };
        let handle = NonNull::new(handle).ok_or_else(|| {
            VitureError::Internal(format!(
                "xr_device_provider_create returned null for product_id={chosen_product_id}"
            ))
        })?;

        let (imu_tx, imu_rx) = unbounded::<IMUReading>();
        let (stereo_tx, stereo_rx) = unbounded::<StereoCameraPacket>();
        let (state_tx, state_rx) = unbounded::<(i32, i32)>();

        callbacks::install_imu_sender(imu_tx);
        callbacks::install_stereo_sender(stereo_tx);
        callbacks::install_state_sender(state_tx);

        Ok(Self {
            handle,
            product_id: chosen_product_id,
            state: parking_lot::Mutex::new(DeviceState::Created),
            connected: AtomicBool::new(false),
            device_type: parking_lot::RwLock::new(DeviceType::Unknown(-1)),
            imu_stream: IMUStream::new(imu_rx),
            depth_stream: DepthStream::new(stereo_rx),
            _state_rx: state_rx,
        })
    }

    pub fn connect(&self) -> Result<()> {
        let (custom_config, cache_dir) = Self::build_initialize_args();
        let custom_ptr = custom_config
            .as_ref()
            .map_or(std::ptr::null(), |v| v.as_ptr());
        let cache_ptr = cache_dir.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());

        check_status("xr_device_provider_initialize", unsafe {
            ffi::xr_device_provider_initialize(self.handle.as_ptr(), custom_ptr, cache_ptr)
        })?;
        *self.state.lock() = DeviceState::Initialized;

        check_status("xr_device_provider_register_state_callback", unsafe {
            ffi::xr_device_provider_register_state_callback(
                self.handle.as_ptr(),
                Some(callbacks::state_callback_ptr()),
            )
        })?;

        let raw_device_type = check_non_negative("xr_device_provider_get_device_type", unsafe {
            ffi::xr_device_provider_get_device_type(self.handle.as_ptr())
        })?;
        let device_type = DeviceType::from(raw_device_type);
        *self.device_type.write() = device_type;

        let mut use_carina_callbacks = matches!(device_type, DeviceType::Carina);
        if let DeviceType::Unknown(_) = device_type {
            let carina_result = unsafe {
                ffi::xr_device_provider_register_callbacks_carina(
                    self.handle.as_ptr(),
                    Some(callbacks::carina_pose_callback_ptr()),
                    Some(callbacks::carina_vsync_callback_ptr()),
                    Some(callbacks::carina_imu_callback_ptr()),
                    Some(callbacks::carina_camera_callback_ptr()),
                )
            };
            if carina_result == 0 {
                use_carina_callbacks = true;
            }
        }

        if use_carina_callbacks {
            check_status("xr_device_provider_register_callbacks_carina", unsafe {
                ffi::xr_device_provider_register_callbacks_carina(
                    self.handle.as_ptr(),
                    Some(callbacks::carina_pose_callback_ptr()),
                    Some(callbacks::carina_vsync_callback_ptr()),
                    Some(callbacks::carina_imu_callback_ptr()),
                    Some(callbacks::carina_camera_callback_ptr()),
                )
            })?;
        } else {
            check_status("xr_device_provider_register_imu_pose_callback", unsafe {
                ffi::xr_device_provider_register_imu_pose_callback(
                    self.handle.as_ptr(),
                    Some(callbacks::imu_pose_callback_ptr()),
                )
            })?;
            let _ = unsafe {
                ffi::xr_device_provider_register_imu_raw_callback(
                    self.handle.as_ptr(),
                    Some(callbacks::imu_raw_callback_ptr()),
                )
            };
        }

        check_status("xr_device_provider_start", unsafe {
            ffi::xr_device_provider_start(self.handle.as_ptr())
        })?;
        *self.state.lock() = DeviceState::Started;

        if !use_carina_callbacks {
            let pose_result = unsafe {
                ffi::xr_device_provider_open_imu(
                    self.handle.as_ptr(),
                    ffi::VITURE_IMU_MODE_POSE,
                    ffi::VITURE_IMU_FREQUENCY_MEDIUM,
                )
            };
            if pose_result != 0 {
                let raw_result = unsafe {
                    ffi::xr_device_provider_open_imu(
                        self.handle.as_ptr(),
                        ffi::VITURE_IMU_MODE_RAW,
                        ffi::VITURE_IMU_FREQUENCY_MEDIUM,
                    )
                };
                if raw_result != 0 && raw_result != -3 {
                    return Err(VitureError::SdkCallFailed {
                        function: "xr_device_provider_open_imu",
                        code: raw_result,
                    });
                }
            }
        }

        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn disconnect(&self) -> Result<()> {
        if !self.connected.load(Ordering::SeqCst) {
            return Ok(());
        }

        let device_type = *self.device_type.read();
        if !matches!(device_type, DeviceType::Carina) {
            let _ = unsafe {
                ffi::xr_device_provider_close_imu(self.handle.as_ptr(), ffi::VITURE_IMU_MODE_POSE)
            };
        }

        check_status("xr_device_provider_stop", unsafe {
            ffi::xr_device_provider_stop(self.handle.as_ptr())
        })?;
        *self.state.lock() = DeviceState::Stopped;

        check_status("xr_device_provider_shutdown", unsafe {
            ffi::xr_device_provider_shutdown(self.handle.as_ptr())
        })?;
        *self.state.lock() = DeviceState::Shutdown;

        self.connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    pub fn imu(&self) -> IMUStream {
        self.imu_stream.clone()
    }

    pub fn depth(&self) -> DepthStream {
        self.depth_stream.clone()
    }

    pub fn device_type(&self) -> DeviceType {
        *self.device_type.read()
    }

    pub fn info(&self) -> Result<DeviceInfo> {
        let mut buf = vec![0u8; 128];
        let mut len = buf.len() as i32;
        check_status("xr_device_provider_get_glasses_version", unsafe {
            ffi::xr_device_provider_get_glasses_version(
                self.handle.as_ptr(),
                buf.as_mut_ptr() as *mut i8,
                &mut len,
            )
        })?;
        let len = len.clamp(0, buf.len() as i32) as usize;
        let version = String::from_utf8_lossy(&buf[..len])
            .trim_end_matches('\0')
            .to_string();
        Ok(DeviceInfo {
            device_type: self.device_type(),
            firmware_version: version,
            product_id: self.product_id,
        })
    }

    pub fn set_brightness(&self, value: u8) -> Result<()> {
        check_status("xr_device_provider_set_brightness_level", unsafe {
            ffi::xr_device_provider_set_brightness_level(self.handle.as_ptr(), value as i32)
        })
    }

    pub fn get_brightness(&self) -> Result<u8> {
        let value = check_non_negative("xr_device_provider_get_brightness_level", unsafe {
            ffi::xr_device_provider_get_brightness_level(self.handle.as_ptr())
        })?;
        Ok(value as u8)
    }

    pub fn set_display_mode(&self, mode: i32) -> Result<()> {
        check_status("xr_device_provider_set_display_mode", unsafe {
            ffi::xr_device_provider_set_display_mode(self.handle.as_ptr(), mode)
        })
    }

    pub fn set_volume(&self, value: u8) -> Result<()> {
        check_status("xr_device_provider_set_volume_level", unsafe {
            ffi::xr_device_provider_set_volume_level(self.handle.as_ptr(), value as i32)
        })
    }

    pub fn read_imu(&self) -> Result<IMUReading> {
        if !self.is_connected() {
            return Err(VitureError::NotConnected);
        }
        self.imu_stream.read()
    }
}

impl Drop for VitureDevice {
    fn drop(&mut self) {
        let _ = self.disconnect();
        callbacks::clear_all();
        unsafe {
            ffi::xr_device_provider_destroy(self.handle.as_ptr());
        }
    }
}
