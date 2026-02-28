import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { getPositionStyle } from '../lib/hud/position';
import { getVideoBounds } from '../lib/hud/sizing';
import type { HudComponentHandle } from '../lib/hud/types';
import { parseInlineStyle } from './utils';

interface VideoProps {
  onVideoEnded?: () => void;
}

const Video = forwardRef<HudComponentHandle, VideoProps>(({ onVideoEnded }, ref) => {
  const [visible, setVisible] = useState(false);
  const [videoPosition, setVideoPosition] = useState('center');
  const [videoWidth, setVideoWidth] = useState('1280px');
  const [videoHeight, setVideoHeight] = useState('720px');
  const [videoOrientation, setVideoOrientation] = useState<'landscape' | 'portrait' | 'square'>('landscape');
  const [videoState, setVideoState] = useState('idle');

  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const aspectRatioRef = useRef(16 / 9);
  const requestedSizeRef = useRef<unknown>('95%');
  const visibleRef = useRef(false);
  const loopRef = useRef(false);
  const volumeRef = useRef(1);
  const srcRef = useRef('');

  const recalculateLayout = useCallback(() => {
    if (!visibleRef.current || typeof window === 'undefined') return;

    const ratio = Number.isFinite(aspectRatioRef.current) && aspectRatioRef.current > 0
      ? aspectRatioRef.current
      : 16 / 9;
    const { maxWidth, maxHeight } = getVideoBounds(requestedSizeRef.current);

    let renderWidth = maxWidth;
    let renderHeight = renderWidth / ratio;

    if (renderHeight > maxHeight) {
      renderHeight = maxHeight;
      renderWidth = renderHeight * ratio;
    }

    if (ratio > 1.05) {
      setVideoOrientation('landscape');
    } else if (ratio < 0.95) {
      setVideoOrientation('portrait');
    } else {
      setVideoOrientation('square');
    }

    setVideoWidth(`${Math.max(1, Math.round(renderWidth))}px`);
    setVideoHeight(`${Math.max(1, Math.round(renderHeight))}px`);
  }, []);

  const stopVideo = useCallback(() => {
    const el = videoElRef.current;
    if (el) {
      el.pause();
      el.removeAttribute('src');
      el.load();
    }
    srcRef.current = '';
    visibleRef.current = false;
    setVisible(false);
    requestedSizeRef.current = '95%';
    aspectRatioRef.current = 16 / 9;
    setVideoOrientation('landscape');
    setVideoState('idle');
  }, []);

  const tickPlay = useCallback(async () => {
    const el = videoElRef.current;
    if (!el || !srcRef.current) return;
    try {
      el.loop = loopRef.current;
      el.volume = volumeRef.current;
      el.muted = volumeRef.current === 0;
      el.src = srcRef.current;
      recalculateLayout();
      await el.play();
      setVideoState('playing');
    } catch {
      setVideoState('error');
    }
  }, [recalculateLayout]);

  useImperativeHandle(ref, () => ({
    async handle(action: string, params: Record<string, any>) {
      if (action === 'play') {
        const src = String(params.src ?? '').trim();
        if (!src) {
          stopVideo();
          setVideoState('error');
          return;
        }

        visibleRef.current = true;
        setVisible(true);
        setVideoState('loading');
        srcRef.current = src;
        setVideoPosition(String(params.position ?? 'center'));
        loopRef.current = Boolean(params.loop);
        volumeRef.current = Math.max(0, Math.min(1, Number(params.volume ?? 1)));
        requestedSizeRef.current = params.size ?? '95%';
        aspectRatioRef.current = 16 / 9;
        recalculateLayout();

        // Wait for next render so the video element is mounted
        requestAnimationFrame(() => {
          tickPlay();
        });
      } else if (action === 'pause') {
        if (visibleRef.current) {
          videoElRef.current?.pause();
          setVideoState('paused');
        }
      } else if (action === 'resume') {
        const el = videoElRef.current;
        if (el) {
          try {
            await el.play();
            setVideoState('playing');
          } catch {
            setVideoState('error');
          }
        }
      } else if (action === 'stop' || action === 'hide') {
        stopVideo();
      } else if (action === 'set_size') {
        requestedSizeRef.current = params.size ?? params;
        recalculateLayout();
      } else if (action === 'set_position') {
        setVideoPosition(String(params.position ?? 'center'));
      } else if (action === 'set_volume') {
        if (typeof params.volume === 'number') {
          volumeRef.current = Math.max(0, Math.min(1, params.volume));
          const el = videoElRef.current;
          if (el) {
            el.volume = volumeRef.current;
            el.muted = volumeRef.current === 0;
          }
        }
      }
    },
    reset() {
      stopVideo();
    },
  }));

  // Window resize handler
  useEffect(() => {
    const onResize = () => {
      if (visibleRef.current) {
        recalculateLayout();
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [recalculateLayout]);

  const onLoadedData = () => {
    if (visibleRef.current && videoState !== 'playing') {
      setVideoState('ready');
    }
  };

  const onLoadedMetadata = () => {
    const el = videoElRef.current;
    if (!el) return;
    const { videoWidth: intrinsicWidth, videoHeight: intrinsicHeight } = el;
    if (intrinsicWidth > 0 && intrinsicHeight > 0) {
      aspectRatioRef.current = intrinsicWidth / intrinsicHeight;
      recalculateLayout();
    }
  };

  const onPlay = () => {
    if (visibleRef.current) {
      setVideoState('playing');
    }
  };

  const onPause = () => {
    if (visibleRef.current && videoState !== 'error') {
      setVideoState('paused');
    }
  };

  const onError = () => {
    setVideoState('error');
  };

  const onEnded = () => {
    if (!loopRef.current) {
      stopVideo();
    }
    onVideoEnded?.();
  };

  if (!visible) return null;

  const styleStr = `${getPositionStyle(videoPosition)}width:${videoWidth};height:${videoHeight};`;

  return (
    <div
      className="media-card video-card"
      data-testid="hud-video-card"
      data-orientation={videoOrientation}
      style={parseInlineStyle(styleStr)}
    >
      <video
        ref={videoElRef}
        className="hud-video"
        data-testid="hud-video"
        data-state={videoState}
        playsInline
        autoPlay
        onEnded={onEnded}
        onLoadedMetadata={onLoadedMetadata}
        onLoadedData={onLoadedData}
        onPlay={onPlay}
        onPause={onPause}
        onError={onError}
      />
    </div>
  );
});

Video.displayName = 'Video';
export default Video;
