const DEFAULT_THROBBER_HZ = 0.1;
const MIN_THROBBER_HZ = 0.05;
const MAX_THROBBER_HZ = 3;

export function clampThrobberHz(input: unknown): number {
  const value = Number(input);
  if (!Number.isFinite(value)) {
    return DEFAULT_THROBBER_HZ;
  }
  return Math.max(MIN_THROBBER_HZ, Math.min(MAX_THROBBER_HZ, value));
}

export function getThrobberFill(elapsedMs: number, hz: number): number {
  const safeHz = clampThrobberHz(hz);
  const radians = (Math.max(0, elapsedMs) / 1000) * safeHz * Math.PI * 2;
  return (Math.sin(radians) + 1) / 2;
}

export function hexToRgb(hex: string): [number, number, number] {
  if (typeof hex !== 'string') return [255, 255, 255];
  const cleaned = hex.replace(/^#/, '');
  if (cleaned.length === 3) {
    const r = parseInt(cleaned[0] + cleaned[0], 16);
    const g = parseInt(cleaned[1] + cleaned[1], 16);
    const b = parseInt(cleaned[2] + cleaned[2], 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
    return [r, g, b];
  }
  if (cleaned.length === 6) {
    const r = parseInt(cleaned.slice(0, 2), 16);
    const g = parseInt(cleaned.slice(2, 4), 16);
    const b = parseInt(cleaned.slice(4, 6), 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
    return [r, g, b];
  }
  return [255, 255, 255];
}

export function getMarbleAnimationValues(elapsedMs: number, hz: number) {
  const safeHz = clampThrobberHz(hz);
  const t = Math.max(0, elapsedMs) / 1000;
  const baseFreq = safeHz * Math.PI * 2;

  // Lissajous curves for fog positions (% of container, range ~20-80%)
  const fogX1 = 50 + 30 * Math.sin(t * baseFreq * 0.7);
  const fogY1 = 50 + 30 * Math.cos(t * baseFreq * 0.5);
  const fogX2 = 50 + 30 * Math.sin(t * baseFreq * 0.5 + Math.PI * 0.75);
  const fogY2 = 50 + 30 * Math.cos(t * baseFreq * 0.3 + Math.PI * 0.5);

  // Sinusoidal glow intensity (0.3 - 1.0)
  const glowIntensity = 0.65 + 0.35 * Math.sin(t * baseFreq);

  // Subtle border-radius variation (48-52%)
  const morphFactor = 50 + 2 * Math.sin(t * baseFreq * 0.4);

  // Gentle vertical bounce (max 2px)
  const bounceY = 2 * Math.sin(t * baseFreq * 0.6);

  return { fogX1, fogY1, fogX2, fogY2, glowIntensity, morphFactor, bounceY };
}
