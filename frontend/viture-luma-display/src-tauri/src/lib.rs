mod audio;
mod audio_devices;
mod config;
mod display;
mod errors;
mod grpc_client;
mod media_store;
mod protocol;
mod state;

use std::borrow::Cow;

use tauri::{Listener, Manager, WebviewUrl, WebviewWindowBuilder};

use crate::audio_devices::{choose_output_device_id, list_output_devices};
use crate::config::DEFAULT_GRPC_ADDRESS;
use crate::errors::Result;
use crate::media_store::MediaStore;
use crate::protocol::AudioOutputsPayload;
use crate::state::AppState;

#[tauri::command]
fn list_audio_outputs(
    state: tauri::State<'_, AppState>,
) -> std::result::Result<AudioOutputsPayload, String> {
    let devices = list_output_devices().map_err(|error| error.to_string())?;
    Ok(AudioOutputsPayload {
        devices,
        selected_device_id: state.selected_audio_device(),
    })
}

#[tauri::command]
fn set_audio_output(
    state: tauri::State<'_, AppState>,
    device_id: Option<String>,
) -> std::result::Result<(), String> {
    state
        .set_selected_audio_device(device_id)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn open_settings_window(app: tauri::AppHandle) -> std::result::Result<(), String> {
    let window = app
        .get_webview_window("settings")
        .ok_or_else(|| "settings window not found".to_string())?;

    window.show().map_err(|error| error.to_string())?;
    window.set_focus().map_err(|error| error.to_string())?;
    Ok(())
}

fn should_open_settings() -> bool {
    std::env::args().any(|arg| arg == "--settings")
}

fn create_hud_window(app: &tauri::AppHandle, monitor: &tauri::Monitor) -> Result<()> {
    let window = WebviewWindowBuilder::new(app, "hud", WebviewUrl::App("/hud".into()))
        .title("VITURE HUD")
        .decorations(true)
        .always_on_top(true)
        .resizable(true)
        .minimizable(true)
        .maximizable(true)
        .position(monitor.position().x as f64, monitor.position().y as f64)
        .inner_size(monitor.size().width as f64, monitor.size().height as f64)
        .build()?;

    window.set_fullscreen(true)?;
    Ok(())
}

fn create_settings_window(app: &tauri::AppHandle, visible: bool) -> Result<()> {
    if app.get_webview_window("settings").is_some() {
        return Ok(());
    }

    let window = WebviewWindowBuilder::new(app, "settings", WebviewUrl::App("/settings".into()))
        .title("VITURE HUD Settings")
        .inner_size(580.0, 360.0)
        .resizable(false)
        .visible(visible)
        .build()?;

    if !visible {
        window.hide()?;
    }

    Ok(())
}

fn run_internal() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let media_store = MediaStore::new();
    let protocol_store = media_store.clone();

    tauri::Builder::default()
        .register_uri_scheme_protocol("hud-media", move |_ctx, request| {
            let path = request.uri().path().trim_start_matches('/');
            match protocol_store.get(path) {
                Some(entry) => http::Response::builder()
                    .status(200)
                    .header("Content-Type", &entry.mime)
                    .header("Access-Control-Allow-Origin", "*")
                    .body(Cow::from(entry.data))
                    .unwrap(),
                None => http::Response::builder()
                    .status(404)
                    .header("Content-Type", "text/plain")
                    .body(Cow::from(b"media not found".as_ref()))
                    .unwrap(),
            }
        })
        .setup(move |app| {
            let runtime_dir = config::ensure_runtime_dir()?;
            let config_path = runtime_dir.join("config.json");
            let loaded_config = config::load_config(&config_path)?;

            let state = AppState::new(config_path, loaded_config.audio_device_id);

            let devices = list_output_devices()?;
            let selected =
                choose_output_device_id(&devices, state.selected_audio_device().as_deref());
            if selected != state.selected_audio_device() {
                state.set_selected_audio_device(selected)?;
            }

            let monitor = display::find_hud_monitor(app.handle())?;
            create_hud_window(app.handle(), &monitor)?;
            create_settings_window(app.handle(), should_open_settings())?;

            let state_for_event = state.clone();
            app.listen("hud:video_ended", move |_| {
                let _ = state_for_event.event_tx().send(protocol::Event::VideoEnded);
            });

            let grpc_app = app.handle().clone();
            let grpc_state = state.clone();
            let grpc_media_store = media_store.clone();
            let grpc_address = DEFAULT_GRPC_ADDRESS.to_string();
            tauri::async_runtime::spawn(async move {
                grpc_client::start(grpc_app, grpc_state, grpc_media_store, grpc_address).await;
            });

            app.manage(state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_audio_outputs,
            set_audio_output,
            open_settings_window
        ])
        .run(tauri::generate_context!())?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    if let Err(error) = run_internal() {
        eprintln!("failed to run viture-hud: {error}");
        std::process::exit(1);
    }
}
