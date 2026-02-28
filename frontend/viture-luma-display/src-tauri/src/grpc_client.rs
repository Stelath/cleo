use tauri::Emitter;
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
const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug)]
struct AssembledImage {
    mime: String,
    position: String,
    duration_ms: u32,
    data: Vec<u8>,
}

#[derive(Debug, Default)]
struct ImageChunkAssembler {
    image_id: Option<String>,
    expected_chunk_index: i32,
    mime: String,
    position: String,
    duration_ms: u32,
    data: Vec<u8>,
}

impl ImageChunkAssembler {
    fn reset(&mut self) {
        self.image_id = None;
        self.expected_chunk_index = 0;
        self.mime.clear();
        self.position.clear();
        self.duration_ms = 0;
        self.data.clear();
    }

    fn ingest(&mut self, chunk: frontend_proto::ImageChunk) -> Result<Option<AssembledImage>> {
        if self.image_id.is_none() {
            self.image_id = Some(chunk.image_id.clone());
            self.expected_chunk_index = 0;
            self.mime = chunk.mime_type.clone();
            self.position = chunk.position.clone();
            self.duration_ms = chunk.duration_ms;
            self.data.clear();
        }

        if self.image_id.as_deref() != Some(chunk.image_id.as_str()) {
            self.reset();
            return Err(AppError::Grpc("image chunk stream switched image_id mid-transfer".to_string()));
        }

        if chunk.chunk_index != self.expected_chunk_index {
            self.reset();
            return Err(AppError::Grpc(format!(
                "image chunk gap: expected {}, got {}",
                self.expected_chunk_index, chunk.chunk_index
            )));
        }

        self.data.extend_from_slice(&chunk.data);
        self.expected_chunk_index += 1;

        if !chunk.is_last {
            return Ok(None);
        }

        let assembled = AssembledImage {
            mime: if self.mime.is_empty() {
                "application/octet-stream".to_string()
            } else {
                self.mime.clone()
            },
            position: self.position.clone(),
            duration_ms: self.duration_ms,
            data: std::mem::take(&mut self.data),
        };
        self.reset();
        Ok(Some(assembled))
    }
}

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
        .map_err(|error| AppError::Grpc(format!("failed to connect: {error}")))?
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);

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
    let mut image_assembler = ImageChunkAssembler::default();

    // Keep a clone of the client for sending user actions
    let mut action_client = client.clone();

    loop {
        tokio::select! {
            message = stream.message() => {
                match message {
                    Ok(Some(display_update)) => {
                        if let Err(error) = handle_display_update(
                            app,
                            state,
                            media_store,
                            &mut image_assembler,
                            display_update,
                        ).await {
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
    image_assembler: &mut ImageChunkAssembler,
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
        DisplayVariant::ImageChunk(chunk) => {
            if let Some(img) = image_assembler.ingest(chunk)? {
                let media_id = media_store.insert(img.mime, img.data);
                let media_url = format!("hud-media://localhost/{media_id}");

                let mut params = serde_json::Map::new();
                params.insert("src".to_string(), serde_json::Value::String(media_url));
                if !img.position.is_empty() {
                    params.insert("position".to_string(), serde_json::Value::String(img.position));
                }
                if img.duration_ms > 0 {
                    params.insert("duration_ms".to_string(), serde_json::json!(img.duration_ms));
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
            state.reset_audio_cancel();
            let device_id = state.selected_audio_device();
            let cancel = state.audio_cancel_token();
            tokio::spawn(async move {
                if let Err(error) =
                    audio::play_pcm_base64(&cmd.data_base64, cmd.sample_rate, device_id, cancel)
                        .await
                {
                    log::error!("audio playback error: {error}");
                }
            });
        }
        DisplayVariant::PlayAudioFile(cmd) => {
            state.reset_audio_cancel();
            let device_id = state.selected_audio_device();
            let cancel = state.audio_cancel_token();
            tokio::spawn(async move {
                if let Err(error) = audio::play_file(&cmd.path, device_id, cancel).await {
                    log::error!("audio file playback error: {error}");
                }
            });
        }
        DisplayVariant::RenderHtml(cmd) => {
            app.emit_to("hud", "hud:render_html", cmd.html)?;
        }
        DisplayVariant::Throbber(req) => {
            if req.visible {
                let mut params = serde_json::Map::new();
                if !req.position.is_empty() {
                    params.insert("position".to_string(), serde_json::Value::String(req.position));
                }
                if !req.color.is_empty() {
                    params.insert("color".to_string(), serde_json::Value::String(req.color));
                }
                if req.hz > 0.0 {
                    params.insert("hz".to_string(), serde_json::json!(req.hz));
                }
                if req.size_px > 0 {
                    params.insert("size_px".to_string(), serde_json::json!(req.size_px));
                }
                app.emit_to(
                    "hud",
                    "hud:command",
                    serde_json::json!({
                        "component": "throbber",
                        "action": "show",
                        "params": params
                    }),
                )?;
            } else {
                app.emit_to(
                    "hud",
                    "hud:command",
                    serde_json::json!({
                        "component": "throbber",
                        "action": "hide",
                        "params": {}
                    }),
                )?;
            }
        }
        DisplayVariant::StopAudio(_) => {
            log::info!("stopping audio playback");
            state.cancel_audio();
        }
        DisplayVariant::AppIndicator(req) => {
            let action = if req.is_active { "activate" } else { "deactivate" };
            let mut params = serde_json::Map::new();
            params.insert("app_name".to_string(), serde_json::Value::String(req.app_name));
            app.emit_to(
                "hud",
                "hud:command",
                serde_json::json!({
                    "component": "app_indicators",
                    "action": action,
                    "params": params
                }),
            )?;
        }
    }

    Ok(())
}
