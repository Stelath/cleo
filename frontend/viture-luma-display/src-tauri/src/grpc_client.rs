use tauri::{Emitter, Manager};
use tokio::sync::broadcast;

use crate::audio;
use crate::errors::{AppError, Result};
use crate::media_store::MediaStore;
use crate::protocol::Event;
use crate::state::AppState;

pub mod frontend_proto {
    tonic::include_proto!("cleo.frontend");
}

use frontend_proto::frontend_service_client::FrontendServiceClient;
use frontend_proto::hud_command::Command as HudCommandVariant;
use frontend_proto::{SubscribeRequest, UserAction, VideoEndedAction};

/// Maximum reconnection backoff delay in seconds.
const MAX_BACKOFF_SECS: u64 = 30;

pub async fn start(
    app: tauri::AppHandle,
    state: AppState,
    media_store: MediaStore,
    grpc_address: String,
) {
    let mut backoff_secs: u64 = 1;

    loop {
        log::info!("connecting to gRPC server at {grpc_address}...");

        match run_session(&app, &state, &media_store, &grpc_address).await {
            Ok(()) => {
                log::info!("gRPC session ended gracefully");
                backoff_secs = 1;
            }
            Err(error) => {
                log::warn!("gRPC session error: {error}");
            }
        }

        log::info!("reconnecting in {backoff_secs}s...");
        tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF_SECS);
    }
}

async fn run_session(
    app: &tauri::AppHandle,
    state: &AppState,
    media_store: &MediaStore,
    grpc_address: &str,
) -> Result<()> {
    let mut client = FrontendServiceClient::connect(grpc_address.to_string())
        .await
        .map_err(|error| AppError::Grpc(format!("failed to connect: {error}")))?;

    log::info!("connected to gRPC server");

    let request = SubscribeRequest {
        client_id: uuid::Uuid::new_v4().to_string(),
    };

    let response = client
        .subscribe_hud_commands(request)
        .await
        .map_err(|error| AppError::Grpc(format!("subscribe failed: {error}")))?;

    let mut stream = response.into_inner();
    let mut event_rx = state.event_tx().subscribe();

    // Keep a clone of the client for sending user actions
    let mut action_client = client.clone();

    loop {
        tokio::select! {
            message = stream.message() => {
                match message {
                    Ok(Some(hud_command)) => {
                        if let Err(error) = handle_hud_command(app, state, media_store, hud_command).await {
                            log::error!("error handling HUD command: {error}");
                        }
                    }
                    Ok(None) => {
                        log::info!("gRPC stream ended by server");
                        return Ok(());
                    }
                    Err(error) => {
                        return Err(AppError::Grpc(format!("stream error: {error}")));
                    }
                }
            }
            event_result = event_rx.recv() => {
                match event_result {
                    Ok(Event::VideoEnded) => {
                        let action = UserAction {
                            action: Some(frontend_proto::user_action::Action::VideoEnded(
                                VideoEndedAction {}
                            )),
                        };
                        if let Err(error) = action_client.send_user_action(action).await {
                            log::warn!("failed to send video_ended action: {error}");
                        }
                    }
                    Ok(_) => {
                        // Other events are not sent back to the server
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => {
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn handle_hud_command(
    app: &tauri::AppHandle,
    state: &AppState,
    media_store: &MediaStore,
    hud_command: frontend_proto::HudCommand,
) -> Result<()> {
    let command = hud_command
        .command
        .ok_or_else(|| AppError::Grpc("received HudCommand with no command variant".to_string()))?;

    match command {
        HudCommandVariant::Component(cmd) => {
            let mut params: serde_json::Value = if cmd.params_json.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&cmd.params_json).unwrap_or_else(|_| {
                    serde_json::Value::Object(serde_json::Map::new())
                })
            };

            // If media_data is present, store it in the MediaStore and inject src
            if !cmd.media_data.is_empty() {
                let mime = if cmd.media_mime.is_empty() {
                    "application/octet-stream".to_string()
                } else {
                    cmd.media_mime
                };
                let media_id = media_store.insert(mime, cmd.media_data);
                let media_url = format!("hud-media://localhost/{media_id}");
                if let Some(obj) = params.as_object_mut() {
                    obj.insert("src".to_string(), serde_json::Value::String(media_url));
                }
            }

            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": cmd.component,
                    "action": cmd.action,
                    "params": params
                }),
            )?;
        }
        HudCommandVariant::PlayAudio(cmd) => {
            let device_id = state.selected_audio_device();
            if let Err(error) =
                audio::play_pcm_base64(&cmd.data_base64, cmd.sample_rate, device_id).await
            {
                log::error!("audio playback error: {error}");
            }
        }
        HudCommandVariant::PlayAudioFile(cmd) => {
            let device_id = state.selected_audio_device();
            if let Err(error) = audio::play_file(&cmd.path, device_id).await {
                log::error!("audio file playback error: {error}");
            }
        }
        HudCommandVariant::RenderHtml(cmd) => {
            app.emit_to("hud", "hud:render_html", cmd.html)?;
        }
        HudCommandVariant::Clear(_) => {
            app.emit_to("hud", "hud:clear", ())?;
        }
    }

    Ok(())
}
