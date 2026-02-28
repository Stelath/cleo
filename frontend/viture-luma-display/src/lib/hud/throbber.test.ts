import { describe, expect, it } from 'vitest';
import { hexToRgb, getMarbleAnimationValues, clampThrobberHz } from './throbber';

describe('hexToRgb', () => {
  it('parses 6-digit hex with hash', () => {
    expect(hexToRgb('#ff8800')).toEqual([255, 136, 0]);
  });

  it('parses 6-digit hex without hash', () => {
    expect(hexToRgb('1a73e8')).toEqual([26, 115, 232]);
  });

  it('parses 3-digit shorthand with hash', () => {
    expect(hexToRgb('#fab')).toEqual([255, 170, 187]);
  });

  it('parses 3-digit shorthand without hash', () => {
    expect(hexToRgb('000')).toEqual([0, 0, 0]);
  });

  it('returns white for empty string', () => {
    expect(hexToRgb('')).toEqual([255, 255, 255]);
  });

  it('returns white for invalid hex', () => {
    expect(hexToRgb('#xyz')).toEqual([255, 255, 255]);
  });

  it('returns white for non-string input', () => {
    expect(hexToRgb(42 as unknown as string)).toEqual([255, 255, 255]);
  });

  it('parses black correctly', () => {
    expect(hexToRgb('#000000')).toEqual([0, 0, 0]);
  });

  it('parses white correctly', () => {
    expect(hexToRgb('#ffffff')).toEqual([255, 255, 255]);
  });
});

describe('getMarbleAnimationValues', () => {
  it('returns all expected keys', () => {
    const result = getMarbleAnimationValues(0, 1);
    expect(result).toHaveProperty('fogX1');
    expect(result).toHaveProperty('fogY1');
    expect(result).toHaveProperty('fogX2');
    expect(result).toHaveProperty('fogY2');
    expect(result).toHaveProperty('glowIntensity');
    expect(result).toHaveProperty('morphFactor');
    expect(result).toHaveProperty('bounceY');
  });

  it('returns centered fog positions at t=0', () => {
    const result = getMarbleAnimationValues(0, 1);
    // At t=0, sin(0)=0 and cos(0)=1, so:
    // fogX1 = 50 + 30*sin(0) = 50
    // fogY1 = 50 + 30*cos(0) = 80
    expect(result.fogX1).toBeCloseTo(50, 5);
    expect(result.fogY1).toBeCloseTo(80, 5);
  });

  it('fog positions stay within 20-80% range', () => {
    for (let ms = 0; ms < 10000; ms += 100) {
      const result = getMarbleAnimationValues(ms, 1);
      expect(result.fogX1).toBeGreaterThanOrEqual(20);
      expect(result.fogX1).toBeLessThanOrEqual(80);
      expect(result.fogY1).toBeGreaterThanOrEqual(20);
      expect(result.fogY1).toBeLessThanOrEqual(80);
      expect(result.fogX2).toBeGreaterThanOrEqual(20);
      expect(result.fogX2).toBeLessThanOrEqual(80);
      expect(result.fogY2).toBeGreaterThanOrEqual(20);
      expect(result.fogY2).toBeLessThanOrEqual(80);
    }
  });

  it('glow intensity stays in 0.3-1.0 range', () => {
    for (let ms = 0; ms < 10000; ms += 100) {
      const result = getMarbleAnimationValues(ms, 1);
      expect(result.glowIntensity).toBeGreaterThanOrEqual(0.3);
      expect(result.glowIntensity).toBeLessThanOrEqual(1.0);
    }
  });

  it('morph factor stays in 48-52% range', () => {
    for (let ms = 0; ms < 10000; ms += 100) {
      const result = getMarbleAnimationValues(ms, 1);
      expect(result.morphFactor).toBeGreaterThanOrEqual(48);
      expect(result.morphFactor).toBeLessThanOrEqual(52);
    }
  });

  it('bounce stays within 2px', () => {
    for (let ms = 0; ms < 10000; ms += 100) {
      const result = getMarbleAnimationValues(ms, 1);
      expect(Math.abs(result.bounceY)).toBeLessThanOrEqual(2);
    }
  });

  it('treats negative elapsed time as zero', () => {
    const result = getMarbleAnimationValues(-1000, 1);
    const resultZero = getMarbleAnimationValues(0, 1);
    expect(result.fogX1).toBeCloseTo(resultZero.fogX1, 10);
    expect(result.glowIntensity).toBeCloseTo(resultZero.glowIntensity, 10);
  });

  it('clamps invalid hz to default', () => {
    const result = getMarbleAnimationValues(1000, NaN);
    const resultDefault = getMarbleAnimationValues(1000, clampThrobberHz(NaN));
    expect(result.fogX1).toBeCloseTo(resultDefault.fogX1, 10);
  });

  it('produces varying values over time', () => {
    const a = getMarbleAnimationValues(0, 1);
    const b = getMarbleAnimationValues(500, 1);
    // Values should differ at different time points
    const same = a.fogX1 === b.fogX1 && a.fogY1 === b.fogY1 && a.glowIntensity === b.glowIntensity;
    expect(same).toBe(false);
  });
});
