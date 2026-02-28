use viture_sensors::sdk::VitureDevice;
use viture_sensors::types::DeviceType;

fn hardware_enabled() -> bool {
    std::env::var("VITURE_HARDWARE").ok().as_deref() == Some("1")
}

#[test]
fn discovers_or_skips_without_hardware() {
    if !hardware_enabled() {
        return;
    }
    let maybe_id = VitureDevice::discover_product_id();
    assert!(maybe_id.is_some(), "No product id discovered");
}

#[test]
fn connect_disconnect_cycle() {
    if !hardware_enabled() {
        return;
    }
    let dev = VitureDevice::new(None).expect("device creation failed");
    dev.connect().expect("connect failed");
    assert!(dev.is_connected());

    let info = dev.info().expect("device info failed");
    assert!(
        !info.firmware_version.is_empty(),
        "firmware version should not be empty"
    );
    assert!(
        matches!(
            info.device_type,
            DeviceType::Gen1 | DeviceType::Gen2 | DeviceType::Carina
        ),
        "unexpected device type: {:?}",
        info.device_type
    );

    let brightness = dev.get_brightness().expect("brightness read failed");
    assert!(
        brightness <= 100,
        "brightness out of expected range: {brightness}"
    );
    dev.set_brightness(brightness)
        .expect("brightness no-op set failed");

    dev.disconnect().expect("disconnect failed");
}
