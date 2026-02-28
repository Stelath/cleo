use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: &str = "1.0.0";

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Command {
    Component {
        component: String,
        action: String,
        #[serde(default)]
        params: serde_json::Value,
    },
    PlayAudio {
        data: String,
        sample_rate: u32,
    },
    PlayAudioFile {
        path: String,
    },
    RenderHtml {
        html: String,
    },
    Clear,
    ListAudioOutputs,
    SetAudioOutput {
        device_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioOutputDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub is_viture_like: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInfo {
    pub selected_device_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutputsPayload {
    pub devices: Vec<AudioOutputDevice>,
    pub selected_device_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Connected {
        protocol_version: String,
        display: DisplayInfo,
        audio: AudioInfo,
    },
    AudioOutputs {
        devices: Vec<AudioOutputDevice>,
        selected_device_id: Option<String>,
    },
    AudioFinished,
    VideoEnded,
    Ack {
        ok: bool,
    },
    Error {
        code: String,
        message: String,
        detail: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_roundtrip_component() {
        let raw =
            r#"{"cmd":"component","component":"toast","action":"show","params":{"text":"ok"}}"#;
        let parsed: Command = serde_json::from_str(raw).expect("parse command");
        match parsed {
            Command::Component {
                component,
                action,
                params,
            } => {
                assert_eq!(component, "toast");
                assert_eq!(action, "show");
                assert_eq!(params["text"], "ok");
            }
            _ => panic!("expected component command"),
        }
    }

    #[test]
    fn event_roundtrip_error() {
        let event = Event::Error {
            code: "bad_command".to_string(),
            message: "invalid".to_string(),
            detail: Some("extra".to_string()),
        };

        let json = serde_json::to_string(&event).expect("serialize event");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse json");

        assert_eq!(parsed["event"], "error");
        assert_eq!(parsed["code"], "bad_command");
    }
}
