import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { getPositionStyle } from '../lib/hud/position';
import { clampThrobberHz, getMarbleAnimationValues, hexToRgb } from '../lib/hud/throbber';
import type { HudComponentHandle } from '../lib/hud/types';
import { parseInlineStyle } from './utils';

function normalizeSize(input: unknown): number {
  const value = Number(input);
  if (!Number.isFinite(value)) {
    return 28;
  }
  return Math.max(16, Math.min(96, Math.round(value)));
}

const Throbber = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [position, setPosition] = useState('top-right');
  const [color, setColor] = useState('#d7ebff');
  const [sizePx, setSizePx] = useState(32);

  const hzRef = useRef(0.1);
  const animFrameRef = useRef(0);
  const animStartRef = useRef(0);
  const visibleRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const glowRgbRef = useRef<[number, number, number]>([215, 235, 255]);

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
    const vals = getMarbleAnimationValues(elapsedMs, hzRef.current);

    const el = containerRef.current;
    if (el) {
      const [r, g, b] = glowRgbRef.current;
      el.style.setProperty('--glow-r', String(r));
      el.style.setProperty('--glow-g', String(g));
      el.style.setProperty('--glow-b', String(b));
      el.style.setProperty('--fog-x1', `${vals.fogX1}%`);
      el.style.setProperty('--fog-y1', `${vals.fogY1}%`);
      el.style.setProperty('--fog-x2', `${vals.fogX2}%`);
      el.style.setProperty('--fog-y2', `${vals.fogY2}%`);
      el.style.setProperty('--glow-intensity', String(vals.glowIntensity));
      el.style.setProperty('--morph', String(vals.morphFactor));
      el.style.setProperty('--bounce-y', String(vals.bounceY));
    }

    animFrameRef.current = requestAnimationFrame(animate);
  }, []);

  const startAnimation = useCallback(() => {
    stopAnimation();
    animStartRef.current = 0;
    animFrameRef.current = requestAnimationFrame(animate);
  }, [stopAnimation, animate]);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'show') {
        visibleRef.current = true;
        setVisible(true);
        setPosition(String(params.position ?? 'top-right'));
        const c = String(params.color ?? '#d7ebff');
        setColor(c);
        glowRgbRef.current = hexToRgb(c);
        hzRef.current = clampThrobberHz(params.hz);
        setSizePx(normalizeSize(params.size_px));
        startAnimation();
      } else if (action === 'set') {
        if (params.position !== undefined) {
          setPosition(String(params.position ?? 'top-right'));
        }
        if (params.color !== undefined) {
          const c = String(params.color ?? '#d7ebff');
          setColor(c);
          glowRgbRef.current = hexToRgb(c);
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
      }
    },
    reset() {
      visibleRef.current = false;
      setVisible(false);
      stopAnimation();
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
      ref={containerRef}
      className="throbber"
      data-testid="hud-throbber"
      style={parseInlineStyle(styleStr)}
    >
      <div className="throbber-core" />
      <div className="throbber-fog throbber-fog-1" />
      <div className="throbber-fog throbber-fog-2" />
      <div className="throbber-glow" />
    </div>
  );
});

Throbber.displayName = 'Throbber';
export default Throbber;
