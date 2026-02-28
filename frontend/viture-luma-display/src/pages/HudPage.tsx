import { useEffect, useRef } from 'react';
import { listen, emit } from '@tauri-apps/api/event';

import Video from '../components/Video';
import Image from '../components/Image';
import HtmlOverlay from '../components/HtmlOverlay';
import Throbber from '../components/Throbber';
import Text from '../components/Text';
import Progress from '../components/Progress';
import Toast from '../components/Toast';
import StatusBar from '../components/StatusBar';
import CardStack from '../components/CardStack';
import type { HudComponentHandle } from '../lib/hud/types';

type HudTestWindow = Window & {
  __HUD_TEST_DISPATCH__?: (payload: Record<string, any>) => void;
};

const getHudTestWindow = () => window as HudTestWindow;

function coerceParams(input: unknown): Record<string, any> {
  if (typeof input === 'object' && input !== null && !Array.isArray(input)) {
    return input as Record<string, any>;
  }
  return {};
}

export default function HudPage() {
  const videoRef = useRef<HudComponentHandle>(null);
  const imageRef = useRef<HudComponentHandle>(null);
  const htmlRef = useRef<HudComponentHandle>(null);
  const throbberRef = useRef<HudComponentHandle>(null);
  const textRef = useRef<HudComponentHandle>(null);
  const progressRef = useRef<HudComponentHandle>(null);
  const toastRef = useRef<HudComponentHandle>(null);
  const statusBarRef = useRef<HudComponentHandle>(null);
  const cardRef = useRef<HudComponentHandle>(null);

  const clearHud = () => {
    toastRef.current?.reset();
    progressRef.current?.reset();
    imageRef.current?.reset();
    videoRef.current?.reset();
    textRef.current?.reset();
    throbberRef.current?.reset();
    htmlRef.current?.reset();
    statusBarRef.current?.reset();
    cardRef.current?.reset();
  };

  const dispatchCommand = (payload: Record<string, any>) => {
    const component = String(payload.component ?? '');
    const action = String(payload.action ?? '');
    const params = coerceParams(payload.params);

    if (component === 'toast') {
      toastRef.current?.handle(action, params);
      return;
    }
    if (component === 'progress') {
      progressRef.current?.handle(action, params);
      return;
    }
    if (component === 'image') {
      imageRef.current?.handle(action, params);
      return;
    }
    if (component === 'video') {
      videoRef.current?.handle(action, params);
      return;
    }
    if (component === 'text') {
      textRef.current?.handle(action, params);
      return;
    }
    if (component === 'throbber') {
      throbberRef.current?.handle(action, params);
      return;
    }
    if (component === 'status_bar') {
      statusBarRef.current?.handle(action, params);
      return;
    }
    if (component === 'card') {
      cardRef.current?.handle(action, params);
      return;
    }
    if (component === 'html') {
      htmlRef.current?.handle(action, params);
      return;
    }
  };

  const onVideoEnded = () => {
    emit('hud:video_ended', {});
  };

  useEffect(() => {
    let unlistenCommand = () => {};
    let unlistenRenderHtml = () => {};
    let unlistenClear = () => {};
    let removeTestBridge = () => {};

    void (async () => {
      unlistenCommand = await listen('hud:command', (event) => {
        dispatchCommand(event.payload as Record<string, any>);
      });

      unlistenRenderHtml = await listen('hud:render_html', (event) => {
        htmlRef.current?.handle('render', { html: event.payload });
      });

      unlistenClear = await listen('hud:clear', () => {
        clearHud();
      });
    })();

    if (import.meta.env.DEV || import.meta.env.MODE === 'test') {
      getHudTestWindow().__HUD_TEST_DISPATCH__ = (payload: Record<string, any>) => {
        dispatchCommand(coerceParams(payload));
      };
      removeTestBridge = () => {
        delete getHudTestWindow().__HUD_TEST_DISPATCH__;
      };
    }

    return () => {
      unlistenCommand();
      unlistenRenderHtml();
      unlistenClear();
      removeTestBridge();
      clearHud();
    };
  }, []);

  return (
    <>
      <title>VITURE HUD</title>
      <div className="hud-root">
        <div className="layer video-layer">
          <Video ref={videoRef} onVideoEnded={onVideoEnded} />
        </div>

        <div className="layer image-layer">
          <Image ref={imageRef} />
        </div>

        <div className="layer html-layer">
          <HtmlOverlay ref={htmlRef} />
        </div>

        <div className="layer overlay-layer">
          <Throbber ref={throbberRef} />
          <Text ref={textRef} />
          <Progress ref={progressRef} />
          <CardStack ref={cardRef} />
        </div>

        <div className="layer toast-layer">
          <Toast ref={toastRef} />
        </div>

        <div className="layer status-layer">
          <StatusBar ref={statusBarRef} />
        </div>
      </div>
    </>
  );
}
