import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import {
  mergeCards,
  normalizeCards,
  normalizeCardPosition,
  normalizeCardWidth,
  normalizeDurationMs,
  removeCard,
  type CardPosition,
  type HudCard,
} from '../lib/hud/cards';
import type { HudComponentHandle } from '../lib/hud/types';
import WeatherCardContent from './WeatherCardContent';

const CardStack = forwardRef<HudComponentHandle>((_props, ref) => {
  const [cards, setCards] = useState<HudCard[]>([]);
  const [cardPosition, setCardPosition] = useState<CardPosition>('right');
  const [cardWidth, setCardWidth] = useState('min(34vw, 460px)');

  const cardCounterRef = useRef(0);
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const nextCardId = useCallback((): string => {
    cardCounterRef.current += 1;
    return `card-${cardCounterRef.current}`;
  }, []);

  const clearHideTimer = useCallback(() => {
    if (hideTimerRef.current) {
      clearTimeout(hideTimerRef.current);
      hideTimerRef.current = null;
    }
  }, []);

  const scheduleHide = useCallback((durationMs: number) => {
    clearHideTimer();
    if (durationMs <= 0) return;
    hideTimerRef.current = setTimeout(() => {
      setCards([]);
      hideTimerRef.current = null;
    }, durationMs);
  }, [clearHideTimer]);

  useImperativeHandle(ref, () => ({
    handle(action: string, params: Record<string, unknown>) {
      if (action === 'show') {
        setCardPosition(normalizeCardPosition(params.position));
        setCardWidth(normalizeCardWidth(params.width));
        setCards(normalizeCards(params, nextCardId));
        scheduleHide(normalizeDurationMs(params.duration_ms));
      } else if (action === 'add') {
        const incoming = normalizeCards(params, nextCardId);
        setCards((prev) => mergeCards(prev, incoming));
        if (params.position !== undefined) {
          setCardPosition(normalizeCardPosition(params.position));
        }
        if (params.width !== undefined) {
          setCardWidth(normalizeCardWidth(params.width));
        }
        if (params.duration_ms !== undefined) {
          scheduleHide(normalizeDurationMs(params.duration_ms));
        }
      } else if (action === 'remove') {
        setCards((prev) => removeCard(prev, params.id));
      } else if (action === 'hide') {
        clearHideTimer();
        setCards([]);
      }
    },
    reset() {
      clearHideTimer();
      setCards([]);
    },
  }));

  useEffect(() => {
    return () => {
      if (hideTimerRef.current) {
        clearTimeout(hideTimerRef.current);
      }
    };
  }, []);

  if (cards.length === 0) return null;

  return (
    <div
      className={`card-stack ${cardPosition}`}
      data-testid="hud-card-stack"
      data-position={cardPosition}
      style={{ '--card-width': cardWidth } as React.CSSProperties}
    >
      {cards.map((card) =>
        card.type === 'weather' ? (
          <WeatherCardContent key={card.id} card={card} />
        ) : (
          <article key={card.id} className="card-item" data-testid="hud-card-item" data-card-id={card.id}>
            {card.imageSrc && (
              <img className="card-image" data-testid="hud-card-image" src={card.imageSrc} alt="" />
            )}

            {card.title && (
              <h3 className="card-title" data-testid="hud-card-title">{card.title}</h3>
            )}

            {card.subtitle && (
              <p className="card-subtitle" data-testid="hud-card-subtitle">{card.subtitle}</p>
            )}

            {card.description && (
              <p className="card-description" data-testid="hud-card-description">{card.description}</p>
            )}

            {card.meta.length > 0 && (
              <dl className="card-meta" data-testid="hud-card-meta">
                {card.meta.map((row, i) => (
                  <div key={i} className="card-meta-row">
                    <dt>{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            )}

            {card.links.length > 0 && (
              <div className="card-links" data-testid="hud-card-links">
                {card.links.map((link, i) => (
                  <a key={i} href={link.url} target="_blank" rel="noreferrer">{link.label}</a>
                ))}
              </div>
            )}
          </article>
        )
      )}
    </div>
  );
});

CardStack.displayName = 'CardStack';
export default CardStack;
