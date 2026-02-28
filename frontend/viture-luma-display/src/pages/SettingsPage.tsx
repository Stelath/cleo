import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type AudioDevice = {
  id: string;
  name: string;
  is_default: boolean;
  is_viture_like: boolean;
};

export default function SettingsPage() {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  const refreshOutputs = async () => {
    setLoading(true);
    setError('');
    setSaved(false);
    try {
      const payload = (await invoke('list_audio_outputs')) as {
        devices: AudioDevice[];
        selected_device_id: string | null;
      };
      setDevices(payload.devices);
      setSelectedDeviceId(payload.selected_device_id ?? '');
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const saveSelection = async () => {
    setSaved(false);
    setError('');
    try {
      await invoke('set_audio_output', {
        device_id: selectedDeviceId ? selectedDeviceId : null,
      });
      setSaved(true);
    } catch (err) {
      setError(String(err));
    }
  };

  useEffect(() => {
    refreshOutputs();
  }, []);

  return (
    <>
      <title>VITURE HUD Settings</title>
      <main className="settings">
        <h1>Audio Output</h1>

        {loading ? (
          <p>Loading devices...</p>
        ) : (
          <>
            <label htmlFor="audio-select">Playback device</label>
            <select
              id="audio-select"
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
            >
              <option value="">System default</option>
              {devices.map((device) => (
                <option key={device.id} value={device.id}>
                  {device.name}
                  {device.is_viture_like ? ' (VITURE)' : ''}
                  {device.is_default ? ' (Default)' : ''}
                </option>
              ))}
            </select>

            <div className="actions">
              <button type="button" onClick={saveSelection}>Save</button>
              <button type="button" className="secondary" onClick={refreshOutputs}>Refresh</button>
            </div>

            {saved && <p className="ok">Saved.</p>}
          </>
        )}

        {error && <p className="error">{error}</p>}
      </main>
    </>
  );
}
