use image::{ImageBuffer, Rgb};
use viture_sensors::camera::RGBCamera;
use viture_sensors::sdk::VitureDevice;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let device = VitureDevice::new(None)?;
    device.connect()?;

    let camera = RGBCamera::new(device.depth());
    let frame = camera.capture()?;
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_vec(frame.width as u32, frame.height as u32, frame.data)
            .ok_or_else(|| anyhow::anyhow!("Invalid frame shape"))?;
    img.save("camera_preview.png")?;
    println!("Saved camera_preview.png");

    device.disconnect()?;
    Ok(())
}
