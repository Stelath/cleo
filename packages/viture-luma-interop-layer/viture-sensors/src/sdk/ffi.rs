#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub type XRDeviceProviderHandle = *mut c_void;

pub type GlassStateCallback =
    Option<unsafe extern "C" fn(glass_state_id: c_int, glass_value: c_int)>;
pub type VitureImuRawCallback =
    Option<unsafe extern "C" fn(data: *mut f32, timestamp: u64, vsync: u64)>;
pub type VitureImuPoseCallback = Option<unsafe extern "C" fn(data: *mut f32, timestamp: u64)>;

pub type XRPoseCallback = Option<unsafe extern "C" fn(pose: *mut f32, timestamp: f64)>;
pub type XRVSyncCallback = Option<unsafe extern "C" fn(timestamp: f64)>;
pub type XRImuCallback = Option<unsafe extern "C" fn(imu: *mut f32, timestamp: f64)>;
pub type XRCameraCallback = Option<
    unsafe extern "C" fn(
        image_left0: *mut c_char,
        image_right0: *mut c_char,
        image_left1: *mut c_char,
        image_right1: *mut c_char,
        timestamp: f64,
        width: c_int,
        height: c_int,
    ),
>;

pub const XR_DEVICE_TYPE_VITURE_GEN1: c_int = 0;
pub const XR_DEVICE_TYPE_VITURE_GEN2: c_int = 1;
pub const XR_DEVICE_TYPE_VITURE_CARINA: c_int = 2;

pub const VITURE_IMU_MODE_RAW: u8 = 0;
pub const VITURE_IMU_MODE_POSE: u8 = 1;
pub const VITURE_IMU_FREQUENCY_LOW: u8 = 0;
pub const VITURE_IMU_FREQUENCY_MEDIUM: u8 = 2;

unsafe extern "C" {
    pub fn xr_device_provider_create(product_id: c_int) -> XRDeviceProviderHandle;
    pub fn xr_device_provider_initialize(
        handle: XRDeviceProviderHandle,
        custom_config: *const c_char,
        cache_file_dir: *const c_char,
    ) -> c_int;
    pub fn xr_device_provider_start(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_stop(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_shutdown(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_destroy(handle: XRDeviceProviderHandle);
    pub fn xr_device_provider_get_thread_id(
        handle: XRDeviceProviderHandle,
        thread_ids: *mut c_int,
    ) -> c_int;
    pub fn xr_device_provider_register_state_callback(
        handle: XRDeviceProviderHandle,
        callback: GlassStateCallback,
    ) -> c_int;
    pub fn xr_device_provider_get_device_type(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_is_product_id_valid(product_id: c_int) -> bool;
    pub fn xr_device_provider_get_market_name(
        product_id: c_int,
        market_name: *mut c_char,
        length: *mut c_int,
    ) -> c_int;
    pub fn xr_device_provider_set_log_level(level: c_int);
    pub fn xr_device_provider_get_log_level() -> c_int;

    pub fn xr_device_provider_register_imu_raw_callback(
        handle: XRDeviceProviderHandle,
        callback: VitureImuRawCallback,
    ) -> c_int;
    pub fn xr_device_provider_register_imu_pose_callback(
        handle: XRDeviceProviderHandle,
        callback: VitureImuPoseCallback,
    ) -> c_int;
    pub fn xr_device_provider_open_imu(
        handle: XRDeviceProviderHandle,
        imu_mode: u8,
        imu_report_frequency: u8,
    ) -> c_int;
    pub fn xr_device_provider_close_imu(handle: XRDeviceProviderHandle, imu_mode: u8) -> c_int;

    pub fn xr_device_provider_register_callbacks_carina(
        handle: XRDeviceProviderHandle,
        pose_callback: XRPoseCallback,
        vsync_callback: XRVSyncCallback,
        imu_callback: XRImuCallback,
        camera_callback: XRCameraCallback,
    ) -> c_int;
    pub fn xr_device_provider_reset_pose_carina(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_get_gl_pose_carina(
        handle: XRDeviceProviderHandle,
        pose: *mut f32,
        predict_time: f64,
        pose_status: *mut c_int,
    ) -> c_int;

    pub fn xr_device_provider_get_film_mode(
        handle: XRDeviceProviderHandle,
        voltage: *mut f32,
    ) -> c_int;
    pub fn xr_device_provider_set_film_mode(handle: XRDeviceProviderHandle, voltage: f32) -> c_int;
    pub fn xr_device_provider_get_duty_cycle(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_set_duty_cycle(
        handle: XRDeviceProviderHandle,
        duty_cycle: c_int,
    ) -> c_int;
    pub fn xr_device_provider_get_display_mode(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_set_display_mode(
        handle: XRDeviceProviderHandle,
        display_mode: c_int,
    ) -> c_int;
    pub fn xr_device_provider_switch_dimension(
        handle: XRDeviceProviderHandle,
        is_3d: bool,
    ) -> c_int;
    pub fn xr_device_provider_get_brightness_level(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_set_brightness_level(
        handle: XRDeviceProviderHandle,
        level: c_int,
    ) -> c_int;
    pub fn xr_device_provider_get_volume_level(handle: XRDeviceProviderHandle) -> c_int;
    pub fn xr_device_provider_set_volume_level(
        handle: XRDeviceProviderHandle,
        level: c_int,
    ) -> c_int;
    pub fn xr_device_provider_get_glasses_version(
        handle: XRDeviceProviderHandle,
        response: *mut c_char,
        length: *mut c_int,
    ) -> c_int;
}
