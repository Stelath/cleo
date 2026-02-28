import { describe, it, expect } from 'vitest';
import {
  APP_ICONS,
  APP_LABELS,
  DEFAULT_ICON,
  getAppIcon,
  getAppLabel,
} from './app-indicators';

const REQUIRED_APPS = ['face_detection', 'navigator', 'recording', 'alzheimers_help'];

describe('APP_ICONS', () => {
  it.each(REQUIRED_APPS)('has an icon mapping for %s', (app) => {
    expect(APP_ICONS[app]).toBeDefined();
    expect(typeof APP_ICONS[app]).toBe('function');
  });
});

describe('APP_LABELS', () => {
  it.each(REQUIRED_APPS)('has a human-readable label for %s', (app) => {
    expect(APP_LABELS[app]).toBeDefined();
    expect(typeof APP_LABELS[app]).toBe('string');
    expect(APP_LABELS[app].length).toBeGreaterThan(0);
  });
});

describe('getAppIcon', () => {
  it('returns the default icon for an unknown app', () => {
    expect(getAppIcon('unknown_app')).toBe(DEFAULT_ICON);
  });

  it('returns the mapped icon for a known app', () => {
    expect(getAppIcon('face_detection')).toBe(APP_ICONS['face_detection']);
  });
});

describe('getAppLabel', () => {
  it('returns the app name as fallback for an unknown app', () => {
    expect(getAppLabel('unknown_app')).toBe('unknown_app');
  });

  it('returns the human-readable label for a known app', () => {
    expect(getAppLabel('navigator')).toBe('Navigator');
  });
});
