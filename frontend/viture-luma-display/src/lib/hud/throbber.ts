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
