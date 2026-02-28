export type VideoBounds = {
  maxWidth: number;
  maxHeight: number;
};

const DEFAULT_SIZE_RATIO = 0.95;
const MIN_SIZE_RATIO = 0.1;
const MAX_SIZE_RATIO = 0.98;
const MIN_DIMENSION_PX = 120;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function viewportSize(): { width: number; height: number } {
  if (typeof window === 'undefined') {
    return { width: 1920, height: 1080 };
  }

  return {
    width: Math.max(1, window.innerWidth || 1),
    height: Math.max(1, window.innerHeight || 1),
  };
}

function parseRequestedSize(input: unknown): { kind: 'ratio' | 'px'; value: number } | null {
  if (typeof input === 'number' && Number.isFinite(input) && input > 0) {
    return input <= 1 ? { kind: 'ratio', value: input } : { kind: 'px', value: input };
  }

  if (typeof input !== 'string') {
    return null;
  }

  const value = input.trim().toLowerCase();
  if (!value) {
    return null;
  }

  if (value.endsWith('%')) {
    const percent = Number(value.slice(0, -1));
    if (Number.isFinite(percent) && percent > 0) {
      return { kind: 'ratio', value: percent / 100 };
    }
    return null;
  }

  if (value.endsWith('px')) {
    const px = Number(value.slice(0, -2));
    if (Number.isFinite(px) && px > 0) {
      return { kind: 'px', value: px };
    }
    return null;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  return numeric <= 1 ? { kind: 'ratio', value: numeric } : { kind: 'px', value: numeric };
}

export function getVideoBounds(requestedSize: unknown): VideoBounds {
  const view = viewportSize();
  const viewportWidth = Math.max(MIN_DIMENSION_PX, Math.round(view.width * MAX_SIZE_RATIO));
  const viewportHeight = Math.max(MIN_DIMENSION_PX, Math.round(view.height * MAX_SIZE_RATIO));

  const parsed = parseRequestedSize(requestedSize);
  if (!parsed) {
    return {
      maxWidth: Math.round(viewportWidth * DEFAULT_SIZE_RATIO),
      maxHeight: Math.round(viewportHeight * DEFAULT_SIZE_RATIO),
    };
  }

  if (parsed.kind === 'ratio') {
    const ratio = clamp(parsed.value, MIN_SIZE_RATIO, MAX_SIZE_RATIO);
    return {
      maxWidth: Math.max(MIN_DIMENSION_PX, Math.round(viewportWidth * ratio)),
      maxHeight: Math.max(MIN_DIMENSION_PX, Math.round(viewportHeight * ratio)),
    };
  }

  const px = Math.round(parsed.value);
  return {
    maxWidth: clamp(px, MIN_DIMENSION_PX, viewportWidth),
    maxHeight: clamp(px, MIN_DIMENSION_PX, viewportHeight),
  };
}
