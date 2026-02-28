import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from 'react';
import { getPositionStyle } from '../../lib/hud/position';
import type { HudComponentHandle } from '../../lib/hud/types';
import { parseInlineStyle } from './utils';

const HudImage = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [src, setSrc] = useState('');
  const [state, setState] = useState('idle');
  const [position, setPosition] = useState('center');
  const [size, setSize] = useState('40%');
  const [opacity, setOpacity] = useState(1);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const hideImage = useCallback((nextState = 'idle') => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setVisible(false);
    setSrc('');
    setState(nextState);
  }, []);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        const newSrc = String(params.src ?? '').trim();
        if (!newSrc) {
          hideImage('error');
          return;
        }

        setState('loading');
        setVisible(true);
        setSrc(newSrc);
        setPosition(String(params.position ?? 'center'));
        setSize(String(params.size ?? '40%'));
        setOpacity(Math.max(0, Math.min(1, Number(params.opacity ?? 1))));

        if (timerRef.current) {
          clearTimeout(timerRef.current);
          timerRef.current = null;
        }

        const durationMs = Number(params.duration_ms ?? 0);
        if (durationMs > 0) {
          timerRef.current = setTimeout(() => {
            hideImage();
          }, durationMs);
        }
      } else if (action === 'hide') {
        hideImage();
      }
    },
    reset() {
      hideImage();
    },
  }));

  const onLoad = () => {
    if (visible) setState('loaded');
  };

  const onError = () => {
    hideImage('error');
  };

  if (!visible) return null;

  const styleStr = `${getPositionStyle(position)}width:${size};opacity:${opacity};`;

  return (
    <div
      className="media-card image-card"
      data-testid="hud-image-card"
      style={parseInlineStyle(styleStr)}
    >
      <img
        className="hud-image"
        data-testid="hud-image"
        data-state={state}
        src={src}
        alt="HUD"
        onLoad={onLoad}
        onError={onError}
      />
    </div>
  );
});

HudImage.displayName = 'HudImage';
export default HudImage;
