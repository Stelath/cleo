import { forwardRef, useImperativeHandle, useState } from 'react';
import type { HudComponentHandle } from '../lib/hud/types';

const StatusBar = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [left, setLeft] = useState('');
  const [center, setCenter] = useState('');
  const [right, setRight] = useState('');

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        setVisible(true);
        setLeft(String(params.left ?? ''));
        setCenter(String(params.center ?? ''));
        setRight(String(params.right ?? ''));
      } else if (action === 'hide') {
        setVisible(false);
      }
    },
    reset() {
      setVisible(false);
    },
  }));

  if (!visible) return null;

  return (
    <div className="status-bar" data-testid="hud-status-bar">
      <div>{left}</div>
      <div>{center}</div>
      <div>{right}</div>
    </div>
  );
});

StatusBar.displayName = 'StatusBar';
export default StatusBar;
