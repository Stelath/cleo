use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct MediaEntry {
    pub mime: String,
    pub data: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct MediaStore {
    entries: Arc<RwLock<HashMap<String, MediaEntry>>>,
}

impl MediaStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store binary media data and return a UUID key.
    pub fn insert(&self, mime: String, data: Vec<u8>) -> String {
        let id = Uuid::new_v4().to_string();
        self.entries
            .write()
            .expect("media store lock poisoned")
            .insert(id.clone(), MediaEntry { mime, data });
        id
    }

    /// Retrieve a stored media entry by UUID.
    pub fn get(&self, id: &str) -> Option<MediaEntry> {
        self.entries
            .read()
            .expect("media store lock poisoned")
            .get(id)
            .cloned()
    }

    /// Remove a stored media entry by UUID.
    #[allow(dead_code)]
    pub fn remove(&self, id: &str) -> Option<MediaEntry> {
        self.entries
            .write()
            .expect("media store lock poisoned")
            .remove(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let store = MediaStore::new();
        let id = store.insert("image/jpeg".to_string(), vec![0xFF, 0xD8, 0xFF]);
        let entry = store.get(&id).expect("entry should exist");
        assert_eq!(entry.mime, "image/jpeg");
        assert_eq!(entry.data, vec![0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn get_missing_returns_none() {
        let store = MediaStore::new();
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn remove_entry() {
        let store = MediaStore::new();
        let id = store.insert("video/mp4".to_string(), vec![0x00]);
        assert!(store.remove(&id).is_some());
        assert!(store.get(&id).is_none());
    }
}
