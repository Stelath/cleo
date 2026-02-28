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

use frontend_proto::display_update::Update as DisplayVariant;
use frontend_proto::frontend_service_client::FrontendServiceClient;
use frontend_proto::{StreamRequest, UserAction, VideoEndedAction};

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

    let request = StreamRequest {
        client_id: uuid::Uuid::new_v4().to_string(),
    };

    let response = client
        .stream_updates(request)
        .await
        .map_err(|error| AppError::Grpc(format!("stream_updates failed: {error}")))?;

    let mut stream = response.into_inner();
    let mut event_rx = state.event_tx().subscribe();

    // Keep a clone of the client for sending user actions
    let mut action_client = client.clone();

    loop {
        tokio::select! {
            message = stream.message() => {
                match message {
                    Ok(Some(display_update)) => {
                        if let Err(error) = handle_display_update(app, state, media_store, display_update).await {
                            log::error!("error handling display update: {error}");
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

async fn handle_display_update(
    app: &tauri::AppHandle,
    state: &AppState,
    media_store: &MediaStore,
    display_update: frontend_proto::DisplayUpdate,
) -> Result<()> {
    let variant = display_update
        .update
        .ok_or_else(|| AppError::Grpc("received DisplayUpdate with no variant".to_string()))?;

    match variant {
        DisplayVariant::Notification(notif) => {
            let mut params = serde_json::Map::new();
            params.insert("title".to_string(), serde_json::Value::String(notif.title));
            params.insert("message".to_string(), serde_json::Value::String(notif.message));
            if !notif.style.is_empty() {
                params.insert("style".to_string(), serde_json::Value::String(notif.style));
            }
            if notif.duration_ms > 0 {
                params.insert("duration_ms".to_string(), serde_json::json!(notif.duration_ms));
            }
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "toast",
                    "action": "show",
                    "params": params
                }),
            )?;
        }
        DisplayVariant::Image(img) => {
            let mime = if img.mime_type.is_empty() {
                "application/octet-stream".to_string()
            } else {
                img.mime_type
            };
            let media_id = media_store.insert(mime, img.data);
            let media_url = format!("hud-media://localhost/{media_id}");

            let mut params = serde_json::Map::new();
            params.insert("src".to_string(), serde_json::Value::String(media_url));
            if !img.position.is_empty() {
                params.insert("position".to_string(), serde_json::Value::String(img.position));
            }
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "image",
                    "action": "show",
                    "params": params
                }),
            )?;
        }
        DisplayVariant::Progress(prog) => {
            let action = if !prog.visible {
                "hide"
            } else if prog.value > 0.0 {
                "set"
            } else {
                "show"
            };
            let mut params = serde_json::Map::new();
            if !prog.label.is_empty() {
                params.insert("label".to_string(), serde_json::Value::String(prog.label));
            }
            params.insert("value".to_string(), serde_json::json!(prog.value));
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "progress",
                    "action": action,
                    "params": params
                }),
            )?;
        }
        DisplayVariant::Text(txt) => {
            let mut params = serde_json::Map::new();
            params.insert("text".to_string(), serde_json::Value::String(txt.text));
            if !txt.position.is_empty() {
                params.insert("position".to_string(), serde_json::Value::String(txt.position));
            }
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "text",
                    "action": "show",
                    "params": params
                }),
            )?;
        }
        DisplayVariant::Card(card_req) => {
            let cards: Vec<serde_json::Value> = card_req
                .cards
                .into_iter()
                .map(|c| {
                    let mut card_json = serde_json::json!({
                        "title": c.title,
                    });
                    let obj = card_json.as_object_mut().unwrap();
                    if !c.subtitle.is_empty() {
                        obj.insert("subtitle".to_string(), serde_json::Value::String(c.subtitle));
                    }
                    if !c.description.is_empty() {
                        obj.insert("description".to_string(), serde_json::Value::String(c.description));
                    }
                    if !c.image_data.is_empty() {
                        let mime = if c.image_mime.is_empty() {
                            "application/octet-stream".to_string()
                        } else {
                            c.image_mime
                        };
                        let media_id = media_store.insert(mime, c.image_data);
                        obj.insert(
                            "image_src".to_string(),
                            serde_json::Value::String(format!("hud-media://localhost/{media_id}")),
                        );
                    }
                    if !c.meta.is_empty() {
                        let meta: serde_json::Map<String, serde_json::Value> = c
                            .meta
                            .into_iter()
                            .map(|kv| (kv.key, serde_json::Value::String(kv.value)))
                            .collect();
                        obj.insert("meta".to_string(), serde_json::Value::Object(meta));
                    }
                    if !c.links.is_empty() {
                        let links: Vec<serde_json::Value> = c
                            .links
                            .into_iter()
                            .map(|l| serde_json::json!({"label": l.label, "url": l.url}))
                            .collect();
                        obj.insert("links".to_string(), serde_json::Value::Array(links));
                    }
                    card_json
                })
                .collect();

            let mut params = serde_json::Map::new();
            params.insert("cards".to_string(), serde_json::Value::Array(cards));
            if !card_req.position.is_empty() {
                params.insert("position".to_string(), serde_json::Value::String(card_req.position));
            }
            if card_req.duration_ms > 0 {
                params.insert("duration_ms".to_string(), serde_json::json!(card_req.duration_ms));
            }
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "card",
                    "action": "show",
                    "params": params
                }),
            )?;
        }
        DisplayVariant::Clear(_) => {
            app.emit_to("hud", "hud:clear", ())?;
        }
        DisplayVariant::PlayAudio(cmd) => {
            let device_id = state.selected_audio_device();
            if let Err(error) =
                audio::play_pcm_base64(&cmd.data_base64, cmd.sample_rate, device_id).await
            {
                log::error!("audio playback error: {error}");
            }
        }
        DisplayVariant::PlayAudioFile(cmd) => {
            let device_id = state.selected_audio_device();
            if let Err(error) = audio::play_file(&cmd.path, device_id).await {
                log::error!("audio file playback error: {error}");
            }
        }
        DisplayVariant::RenderHtml(cmd) => {
            app.emit_to("hud", "hud:render_html", cmd.html)?;
        }
    }

    Ok(())
}
