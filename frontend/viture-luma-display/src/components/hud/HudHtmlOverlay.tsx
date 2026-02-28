import { forwardRef, useImperativeHandle, useState } from 'react';
import type { HudComponentHandle } from '../../lib/hud/types';

const HudHtmlOverlay = forwardRef<HudComponentHandle>((_props, ref) => {
  const [visible, setVisible] = useState(false);
  const [htmlContent, setHtmlContent] = useState('');

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

  if (!visible) return null;

  return (
    <div
      className="hud-html"
      data-testid="hud-html-overlay"
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
});

HudHtmlOverlay.displayName = 'HudHtmlOverlay';
export default HudHtmlOverlay;
