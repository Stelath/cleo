import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { getPositionStyle } from '../lib/hud/position';
import { clampThrobberHz, getThrobberFill } from '../lib/hud/throbber';
import type { HudComponentHandle } from '../lib/hud/types';
import { parseInlineStyle } from './utils';

function normalizeSize(input: unknown): number {
  const value = Number(input);
  if (!Number.isFinite(value)) {
    return 28;
  }
  return Math.max(16, Math.min(96, Math.round(value)));
}

const HudThrobber = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState('top-right');
  const [color, setColor] = useState('#ffffff');
  const [sizePx, setSizePx] = useState(32);
  const [fill, setFill] = useState(0);

  const hzRef = useRef(0.1);
  const animFrameRef = useRef(0);
  const animStartRef = useRef(0);
  const visibleRef = useRef(false);

  const stopAnimation = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = 0;
    }
  }, []);

  const animate = useCallback((nowMs: number) => {
    if (!visibleRef.current) {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = 0;
      }
      return;
    }

    if (animStartRef.current === 0) {
      animStartRef.current = nowMs;
    }

    const elapsedMs = nowMs - animStartRef.current;
    setFill(getThrobberFill(elapsedMs, hzRef.current));
    animFrameRef.current = requestAnimationFrame(animate);
  }, []);

  const startAnimation = useCallback(() => {
    stopAnimation();
    animStartRef.current = 0;
    setFill(0);
    animFrameRef.current = requestAnimationFrame(animate);
  }, [stopAnimation, animate]);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        visibleRef.current = true;
        setVisible(true);
        setPosition(String(params.position ?? 'top-right'));
        setColor(String(params.color ?? '#d7ebff'));
        hzRef.current = clampThrobberHz(params.hz);
        setSizePx(normalizeSize(params.size_px));
        startAnimation();
      } else if (action === 'set') {
        if (params.position !== undefined) {
          setPosition(String(params.position ?? 'top-right'));
        }
        if (params.color !== undefined) {
          setColor(String(params.color ?? '#ffffff'));
        }
        if (params.hz !== undefined) {
          hzRef.current = clampThrobberHz(params.hz);
        }
        if (params.size_px !== undefined) {
          setSizePx(normalizeSize(params.size_px));
        }
      } else if (action === 'hide') {
        visibleRef.current = false;
        setVisible(false);
        stopAnimation();
        setFill(0);
      }
    },
    reset() {
      visibleRef.current = false;
      setVisible(false);
      stopAnimation();
      setFill(0);
    },
  }));

  useEffect(() => {
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = 0;
      }
    };
  }, []);

  if (!visible) return null;

  const styleStr = `${getPositionStyle(position)}color:${color};width:${sizePx}px;height:${sizePx}px;`;

  return (
    <div
      className="throbber"
      data-testid="hud-throbber"
      data-fill={fill}
      style={parseInlineStyle(styleStr)}
    >
      <div
        className="throbber-fill"
        style={{ transform: `translate(-50%,-50%) scale(${Math.max(0.001, fill)})` }}
      />
    </div>
  );
});

HudThrobber.displayName = 'HudThrobber';
export default HudThrobber;
