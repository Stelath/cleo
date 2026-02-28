import type React from 'react';

/**
 * Converts a CSS inline style string (e.g. "top:50%;left:50%;transform:translate(-50%,-50%);")
 * into a React CSSProperties object.
 */
export function parseInlineStyle(cssString: string): React.CSSProperties {
  const style: Record<string, string> = {};
  cssString.split(';').forEach((part) => {
    const trimmed = part.trim();
    if (!trimmed) return;
    const colonIndex = trimmed.indexOf(':');
    if (colonIndex < 0) return;
    const key = trimmed.slice(0, colonIndex).trim();
    const value = trimmed.slice(colonIndex + 1).trim();
    const camelKey = key.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    style[camelKey] = value;
  });
  return style as React.CSSProperties;
}
