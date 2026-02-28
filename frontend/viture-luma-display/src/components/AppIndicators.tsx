import { forwardRef, useImperativeHandle, useState } from 'react';
import type { HudComponentHandle } from '../lib/hud/types';
import { getAppIcon, getAppLabel } from '../lib/hud/app-indicators';

const AppIndicators = forwardRef<HudComponentHandle>((_props, ref) => {
  const [activeApps, setActiveApps] = useState<Set<string>>(new Set());

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, any>) {
      const appName = String(params.app_name ?? '');
      if (!appName) return;

      if (action === 'activate') {
        setActiveApps((prev) => {
          const next = new Set(prev);
          next.add(appName);
          return next;
        });
      } else if (action === 'deactivate') {
        setActiveApps((prev) => {
          const next = new Set(prev);
          next.delete(appName);
          return next;
        });
      }
    },
    reset() {
      setActiveApps(new Set());
    },
  }));

  if (activeApps.size === 0) return null;

  const sorted = [...activeApps].sort();

  return (
    <div className="app-indicators" data-testid="hud-app-indicators">
      {sorted.map((appName) => {
        const Icon = getAppIcon(appName);
        return (
          <div
            key={appName}
            className="app-indicator-pill"
            data-testid={`app-indicator-${appName}`}
            title={getAppLabel(appName)}
          >
            <Icon className="app-indicator-icon" />
          </div>
        );
      })}
    </div>
  );
});

AppIndicators.displayName = 'AppIndicators';
export default AppIndicators;
