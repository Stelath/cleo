import type { IconType } from 'react-icons';
import {
  LuScanFace,
  LuNavigation,
  LuCircleDot,
  LuBrain,
  LuActivity,
} from 'react-icons/lu';

export const APP_ICONS: Record<string, IconType> = {
  face_detection: LuScanFace,
  navigator: LuNavigation,
  recording: LuCircleDot,
  alzheimers_help: LuBrain,
};

export const APP_LABELS: Record<string, string> = {
  face_detection: 'Face Detection',
  navigator: 'Navigator',
  recording: 'Recording',
  alzheimers_help: "Alzheimer's Help",
};

export const DEFAULT_ICON: IconType = LuActivity;

export function getAppIcon(appName: string): IconType {
  return APP_ICONS[appName] ?? DEFAULT_ICON;
}

export function getAppLabel(appName: string): string {
  return APP_LABELS[appName] ?? appName;
}
