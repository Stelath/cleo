use cpal::traits::{DeviceTrait, HostTrait};

use crate::errors::{AppError, Result};
use crate::protocol::AudioOutputDevice;

fn normalize_device_name(name: &str) -> String {
    name.trim().to_string()
}

pub fn is_viture_like(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    lowered.contains("viture") || lowered.contains("xr glasses")
}

pub fn list_output_devices() -> Result<Vec<AudioOutputDevice>> {
    let host = cpal::default_host();
    let default_name = host
        .default_output_device()
        .and_then(|device| device.name().ok())
        .map(|name| normalize_device_name(&name));

    let mut devices = Vec::new();
    let output_devices = host
        .output_devices()
        .map_err(|error| AppError::Audio(format!("failed to enumerate output devices: {error}")))?;

    for device in output_devices {
        let Ok(raw_name) = device.name() else {
            continue;
        };
        let name = normalize_device_name(&raw_name);
        devices.push(AudioOutputDevice {
            id: name.clone(),
            name: name.clone(),
            is_default: default_name.as_deref() == Some(name.as_str()),
            is_viture_like: is_viture_like(&name),
        });
    }

    devices.sort_by(|a, b| {
        b.is_viture_like
            .cmp(&a.is_viture_like)
            .then(b.is_default.cmp(&a.is_default))
            .then(a.name.cmp(&b.name))
    });

    Ok(devices)
}

pub fn choose_output_device_id(
    devices: &[AudioOutputDevice],
    preferred: Option<&str>,
) -> Option<String> {
    if let Some(preferred_id) = preferred {
        if devices.iter().any(|device| device.id == preferred_id) {
            return Some(preferred_id.to_string());
        }
    }

    if let Some(device) = devices.iter().find(|device| device.is_viture_like) {
        return Some(device.id.clone());
    }

    if let Some(device) = devices.iter().find(|device| device.is_default) {
        return Some(device.id.clone());
    }

    devices.first().map(|device| device.id.clone())
}

pub fn find_output_device(device_id: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();

    if let Some(id) = device_id {
        let output_devices = host.output_devices().map_err(|error| {
            AppError::Audio(format!("failed to enumerate output devices: {error}"))
        })?;

        for device in output_devices {
            if let Ok(name) = device.name() {
                if normalize_device_name(&name) == id {
                    return Ok(device);
                }
            }
        }

        return Err(AppError::Audio(format!(
            "audio output device not found: {id}"
        )));
    }

    host.default_output_device()
        .ok_or_else(|| AppError::Audio("no default output device available".to_string()))
}

#[cfg(test)]
mod tests {
    use crate::protocol::AudioOutputDevice;

    use super::{choose_output_device_id, is_viture_like};

    #[test]
    fn detects_viture_like_names() {
        assert!(is_viture_like("VITURE XR Glasses"));
        assert!(is_viture_like("My XR Glasses Output"));
        assert!(!is_viture_like("Internal Speakers"));
    }

    #[test]
    fn prefers_configured_device_when_available() {
        let devices = vec![
            AudioOutputDevice {
                id: "A".to_string(),
                name: "A".to_string(),
                is_default: false,
                is_viture_like: false,
            },
            AudioOutputDevice {
                id: "B".to_string(),
                name: "B".to_string(),
                is_default: true,
                is_viture_like: true,
            },
        ];

        let chosen = choose_output_device_id(&devices, Some("A"));
        assert_eq!(chosen.as_deref(), Some("A"));
    }

    #[test]
    fn falls_back_to_viture_then_default() {
        let devices = vec![
            AudioOutputDevice {
                id: "Default".to_string(),
                name: "Default".to_string(),
                is_default: true,
                is_viture_like: false,
            },
            AudioOutputDevice {
                id: "Viture".to_string(),
                name: "Viture".to_string(),
                is_default: false,
                is_viture_like: true,
            },
        ];

        let chosen = choose_output_device_id(&devices, None);
        assert_eq!(chosen.as_deref(), Some("Viture"));
    }
}
