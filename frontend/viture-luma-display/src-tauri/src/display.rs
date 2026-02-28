use crate::errors::{AppError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
struct MonitorCandidate {
    name: Option<String>,
    width: u32,
    height: u32,
    is_primary: bool,
}

fn is_viture_name(name: &str) -> bool {
    name.to_ascii_lowercase().contains("viture")
}

fn choose_monitor_index(monitors: &[MonitorCandidate]) -> Option<usize> {
    if monitors.is_empty() {
        return None;
    }

    if let Some(index) = monitors
        .iter()
        .position(|monitor| monitor.name.as_deref().is_some_and(is_viture_name))
    {
        return Some(index);
    }

    if let Some(index) = monitors
        .iter()
        .position(|monitor| monitor.width == 1920 && monitor.height == 1080 && !monitor.is_primary)
    {
        return Some(index);
    }

    if let Some(index) = monitors.iter().position(|monitor| !monitor.is_primary) {
        return Some(index);
    }

    Some(0)
}

pub fn find_hud_monitor(app: &tauri::AppHandle) -> Result<tauri::Monitor> {
    let monitors = app.available_monitors()?;
    if monitors.is_empty() {
        return Err(AppError::DisplayNotFound);
    }

    let primary_monitor = app.primary_monitor()?;

    let candidates: Vec<MonitorCandidate> = monitors
        .iter()
        .map(|monitor| {
            let is_primary = primary_monitor.as_ref().is_some_and(|primary| {
                primary.position() == monitor.position()
                    && primary.size() == monitor.size()
                    && primary.name() == monitor.name()
            });

            MonitorCandidate {
                name: monitor.name().cloned(),
                width: monitor.size().width,
                height: monitor.size().height,
                is_primary,
            }
        })
        .collect();

    let selected_index = choose_monitor_index(&candidates).ok_or(AppError::DisplayNotFound)?;

    monitors
        .into_iter()
        .nth(selected_index)
        .ok_or(AppError::DisplayNotFound)
}

pub fn display_info(monitor: &tauri::Monitor) -> crate::protocol::DisplayInfo {
    crate::protocol::DisplayInfo {
        name: monitor
            .name()
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string()),
        width: monitor.size().width,
        height: monitor.size().height,
    }
}

#[cfg(test)]
mod tests {
    use super::{choose_monitor_index, MonitorCandidate};

    fn monitor(name: Option<&str>, width: u32, height: u32, is_primary: bool) -> MonitorCandidate {
        MonitorCandidate {
            name: name.map(ToString::to_string),
            width,
            height,
            is_primary,
        }
    }

    #[test]
    fn chooses_viture_monitor_first() {
        let monitors = vec![
            monitor(Some("Built-in Display"), 3024, 1964, true),
            monitor(Some("VITURE XR Glasses"), 1920, 1080, false),
        ];

        let index = choose_monitor_index(&monitors).expect("selected index");
        assert_eq!(index, 1);
    }

    #[test]
    fn chooses_non_primary_1080p_when_no_named_viture() {
        let monitors = vec![
            monitor(Some("Primary"), 2560, 1440, true),
            monitor(Some("External"), 1920, 1080, false),
        ];

        let index = choose_monitor_index(&monitors).expect("selected index");
        assert_eq!(index, 1);
    }

    #[test]
    fn falls_back_to_primary_if_single_monitor() {
        let monitors = vec![monitor(Some("Only"), 2560, 1440, true)];
        let index = choose_monitor_index(&monitors).expect("selected index");
        assert_eq!(index, 0);
    }
}
