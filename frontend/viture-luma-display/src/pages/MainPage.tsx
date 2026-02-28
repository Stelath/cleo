import { invoke } from '@tauri-apps/api/core';

export default function MainPage() {
  const openSettings = async () => {
    await invoke('open_settings_window');
  };

  return (
    <main className="main-page">
      <h1>VITURE HUD</h1>
      <p>
        HUD window runs at <code>/hud</code>. Use websocket triggers on{' '}
        <code>ws://127.0.0.1:9876</code>.
      </p>
      <button type="button" onClick={openSettings}>
        Open Settings Window
      </button>
    </main>
  );
}
