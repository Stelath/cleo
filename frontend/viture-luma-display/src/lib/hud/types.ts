export interface HudComponentHandle {
  handle(action: string, params: Record<string, any>): void | Promise<void>;
  reset(): void;
}
