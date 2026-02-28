const POSITION_STYLES: Record<string, string> = {
  center: 'top:50%;left:50%;transform:translate(-50%,-50%);',
  'top-center': 'top:6%;left:50%;transform:translateX(-50%);',
  top: 'top:6%;left:50%;transform:translateX(-50%);',
  'bottom-center': 'bottom:8%;left:50%;transform:translateX(-50%);',
  bottom: 'bottom:8%;left:50%;transform:translateX(-50%);',
  left: 'top:50%;left:4%;transform:translateY(-50%);',
  right: 'top:50%;right:4%;transform:translateY(-50%);',
  'center-left': 'top:50%;left:4%;transform:translateY(-50%);',
  'center-right': 'top:50%;right:4%;transform:translateY(-50%);',
  'top-left': 'top:6%;left:4%;',
  'top-right': 'top:6%;right:4%;',
  'upper-right': 'top:15%;right:4%;',
  'bottom-left': 'bottom:8%;left:4%;',
  'bottom-right': 'bottom:8%;right:4%;',
};

function normalizePosition(position: unknown): string {
  if (typeof position !== 'string') {
    return 'center';
  }
  const normalized = position.trim().toLowerCase().replace(/[\s_]+/g, '-');
  return normalized || 'center';
}

export function getPositionStyle(position: unknown): string {
  const key = normalizePosition(position);
  return POSITION_STYLES[key] ?? POSITION_STYLES.center;
}
