import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import type { HudComponentHandle } from '../lib/hud/types';

const HtmlOverlay = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [htmlContent, setHtmlContent] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      if (action === 'render' || action === 'show') {
        setVisible(true);
        setHtmlContent(String(params.html ?? ''));
      } else if (action === 'hide') {
        setVisible(false);
        setHtmlContent('');
      }
    },
    reset() {
      setVisible(false);
      setHtmlContent('');
    },
  }));

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const videos = el.querySelectorAll('video');
    if (videos.length === 0) return;

    const onEnded = () => {
      setVisible(false);
      setHtmlContent('');
    };
    videos.forEach((v) => v.addEventListener('ended', onEnded));
    return () => {
      videos.forEach((v) => v.removeEventListener('ended', onEnded));
    };
  }, [htmlContent]);

  if (!visible) return null;

  return (
    <div
      ref={containerRef}
      className="hud-html"
      data-testid="hud-html-overlay"
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
});

HtmlOverlay.displayName = 'HtmlOverlay';
export default HtmlOverlay;
