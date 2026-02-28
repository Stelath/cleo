import { forwardRef, useImperativeHandle, useState } from 'react';
import { getPositionStyle } from '../lib/hud/position';
import type { HudComponentHandle } from '../lib/hud/types';
import { parseInlineStyle } from './utils';

const Progress = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [style, setStyle] = useState<'bar' | 'ring'>('bar');
  const [position, setPosition] = useState('bottom-center');
  const [color, setColor] = useState('#6bd3ff');
  const [value, setValue] = useState(0);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        setVisible(true);
        setStyle(params.style === 'ring' ? 'ring' : 'bar');
        setPosition(String(params.position ?? 'bottom-center'));
        setColor(String(params.color ?? '#6bd3ff'));
      } else if (action === 'set') {
        const v = Number(params.value ?? 0);
        setValue(Math.max(0, Math.min(1, v)));
      } else if (action === 'hide') {
        setVisible(false);
      }
    },
    reset() {
      setVisible(false);
    },
  }));

  if (!visible) return null;

  const circumference = 2 * Math.PI * 44;

  if (style === 'ring') {
    return (
      <div
        className="progress-ring-wrap"
        data-testid="hud-progress"
        data-value={value}
        style={parseInlineStyle(getPositionStyle(position))}
      >
        <svg className="ring" viewBox="0 0 100 100" data-testid="hud-progress-ring">
          <circle className="ring-bg" cx="50" cy="50" r="44" />
          <circle
            className="ring-fg"
            cx="50"
            cy="50"
            r="44"
            style={{
              stroke: color,
              strokeDasharray: circumference,
              strokeDashoffset: (1 - value) * circumference,
            }}
          />
        </svg>
      </div>
    );
  }

  return (
    <div
      className="progress-wrap"
      data-testid="hud-progress"
      style={parseInlineStyle(getPositionStyle(position))}
    >
      <div className="progress-track">
        <div
          className="progress-fill"
          data-testid="hud-progress-fill"
          data-value={value}
          style={{
            width: `${value * 100}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
});

Progress.displayName = 'Progress';
export default Progress;
