import { forwardRef, useImperativeHandle, useState } from 'react';
import { getPositionStyle } from '../lib/hud/position';
import type { HudComponentHandle } from '../lib/hud/types';
import { parseInlineStyle } from './utils';

const Text = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [content, setContent] = useState('');
  const [position, setPosition] = useState('center');

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        setVisible(true);
        setContent(String(params.text ?? ''));
        setPosition(String(params.position ?? 'center'));
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
    <div
      className="hud-text"
      data-testid="hud-text-overlay"
      style={parseInlineStyle(getPositionStyle(position))}
    >
      {content}
    </div>
  );
});

Text.displayName = 'Text';
export default Text;
