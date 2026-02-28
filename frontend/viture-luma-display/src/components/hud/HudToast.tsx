import { forwardRef, useCallback, useImperativeHandle, useRef, useState } from 'react';
import { getPositionStyle } from '../../lib/hud/position';
import type { HudComponentHandle } from '../../lib/hud/types';
import { parseInlineStyle } from './utils';

type ToastItem = {
  id: number;
  title: string;
  message: string;
  style: string;
  position: string;
};

const HudToast = forwardRef<HudComponentHandle>((_props, ref) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const toastIdRef = useRef(0);

  const removeToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((item) => item.id !== id));
  }, []);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        const title = String(params.title ?? params.heading ?? 'VITURE HUD').trim() || 'VITURE HUD';
        const message = String(params.message ?? params.text ?? '').trim();
        const rawStyle = String(params.style ?? 'info');
        const style = ['info', 'success', 'warning', 'error'].includes(rawStyle) ? rawStyle : 'info';
        const position = String(params.position ?? 'top-right');
        const durationMs = Number(params.duration_ms ?? 3000);

        toastIdRef.current += 1;
        const id = toastIdRef.current;
        const newToast: ToastItem = { id, title, message, style, position };

        setToasts((prev) => [...prev, newToast].slice(-5));

        setTimeout(() => {
          removeToast(id);
        }, durationMs);
      } else if (action === 'hide') {
        setToasts([]);
      }
    },
    reset() {
      setToasts([]);
    },
  }));

  if (toasts.length === 0) return null;

  const lastPosition = toasts[toasts.length - 1].position;

  return (
    <div
      className="toast-stack"
      data-testid="hud-toast-container"
      style={parseInlineStyle(getPositionStyle(lastPosition))}
    >
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.style}`} data-testid="hud-toast-item">
          <div className="toast-title" data-testid="hud-toast-title">{toast.title}</div>
          {toast.message && (
            <div className="toast-message" data-testid="hud-toast-message">{toast.message}</div>
          )}
        </div>
      ))}
    </div>
  );
});

HudToast.displayName = 'HudToast';
export default HudToast;
