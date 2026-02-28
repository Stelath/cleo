import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HudPage from './pages/HudPage';
import SettingsPage from './pages/SettingsPage';
import MainPage from './pages/MainPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/hud" element={<HudPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
