export type CardPosition = 'left' | 'center' | 'right';

export type HudCardMetaRow = {
  label: string;
  value: string;
};

export type HudCardLink = {
  label: string;
  url: string;
};

export type HudCard = {
  id: string;
  type: string;
  title: string;
  subtitle: string;
  description: string;
  imageSrc: string;
  meta: HudCardMetaRow[];
  links: HudCardLink[];
};

const DEFAULT_CARD_WIDTH = 'min(34vw, 460px)';
const MIN_CARD_WIDTH_PX = 220;
const MAX_CARD_WIDTH_PX = 920;
const MAX_DURATION_MS = 300000;

function asTrimmedString(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }
  if (value === null || value === undefined) {
    return '';
  }
  return String(value).trim();
}

function normalizeMeta(meta: unknown): HudCardMetaRow[] {
  if (Array.isArray(meta)) {
    return meta
      .map((entry) => {
        if (typeof entry !== 'object' || entry === null) {
          return null;
        }
        const row = entry as Record<string, unknown>;
        const label = asTrimmedString(row.label ?? row.key);
        const value = asTrimmedString(row.value);
        if (!label || !value) {
          return null;
        }
        return { label, value };
      })
      .filter((entry): entry is HudCardMetaRow => entry !== null);
  }

  if (typeof meta === 'object' && meta !== null) {
    return Object.entries(meta as Record<string, unknown>)
      .map(([label, value]) => ({ label: asTrimmedString(label), value: asTrimmedString(value) }))
      .filter((entry) => entry.label && entry.value);
  }

  return [];
}

function normalizeLinks(links: unknown): HudCardLink[] {
  if (!Array.isArray(links)) {
    return [];
  }

  return links
    .map((entry) => {
      if (typeof entry !== 'object' || entry === null) {
        return null;
      }
      const row = entry as Record<string, unknown>;
      const url = asTrimmedString(row.url);
      if (!url) {
        return null;
      }
      const label = asTrimmedString(row.label) || url;
      return { label, url };
    })
    .filter((entry): entry is HudCardLink => entry !== null);
}

function normalizeSingleCard(value: unknown, nextCardId: () => string): HudCard | null {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const raw = value as Record<string, unknown>;
  const title = asTrimmedString(raw.title);
  const subtitle = asTrimmedString(raw.subtitle);
  const description = asTrimmedString(raw.description);
  const imageSrc = asTrimmedString(raw.image_src ?? raw.imageSrc ?? raw.image);
  const meta = normalizeMeta(raw.meta);
  const links = normalizeLinks(raw.links);

  const hasVisibleContent = Boolean(title || subtitle || description || imageSrc || meta.length || links.length);
  if (!hasVisibleContent) {
    return null;
  }

  const id = asTrimmedString(raw.id) || nextCardId();

  // Card type can be set explicitly via raw.type, or via a __card_type
  // convention in the meta array (used when the backend sends type info
  // through the generic KeyValue meta field of the proto Card message).
  let type = asTrimmedString(raw.type);
  if (!type) {
    const cardTypeMeta = meta.find((m) => m.label === '__card_type');
    if (cardTypeMeta) {
      type = cardTypeMeta.value;
    }
  }

  // Filter __card_type out of the displayed meta rows
  const displayMeta = meta.filter((m) => m.label !== '__card_type');

  return {
    id,
    type,
    title,
    subtitle,
    description,
    imageSrc,
    meta: displayMeta,
    links,
  };
}

export function normalizeCards(
  params: Record<string, unknown>,
  nextCardId: () => string,
): HudCard[] {
  const rawCards = Array.isArray(params.cards)
    ? params.cards
    : params.card !== undefined
      ? [params.card]
      : [params];

  return rawCards
    .map((entry) => normalizeSingleCard(entry, nextCardId))
    .filter((entry): entry is HudCard => entry !== null);
}

export function mergeCards(current: HudCard[], incoming: HudCard[]): HudCard[] {
  if (incoming.length === 0) {
    return current;
  }

  const byId = new Map<string, HudCard>();
  for (const card of current) {
    byId.set(card.id, card);
  }

  for (const card of incoming) {
    byId.set(card.id, card);
  }

  const result: HudCard[] = [];
  for (const card of current) {
    const updated = byId.get(card.id);
    if (updated) {
      result.push(updated);
      byId.delete(card.id);
    }
  }

  for (const card of incoming) {
    const remaining = byId.get(card.id);
    if (remaining) {
      result.push(remaining);
      byId.delete(card.id);
    }
  }

  return result;
}

export function removeCard(cards: HudCard[], id: unknown): HudCard[] {
  const normalizedId = asTrimmedString(id);
  if (!normalizedId) {
    return cards;
  }
  return cards.filter((card) => card.id !== normalizedId);
}

export function normalizeCardPosition(input: unknown): CardPosition {
  const value = asTrimmedString(input).toLowerCase().replace(/[\s_]+/g, '-');
  if (value.includes('left')) {
    return 'left';
  }
  if (value.includes('right')) {
    return 'right';
  }
  return 'center';
}

export function normalizeCardWidth(input: unknown): string {
  if (typeof input === 'number' && Number.isFinite(input) && input > 0) {
    const px = Math.round(Math.max(MIN_CARD_WIDTH_PX, Math.min(MAX_CARD_WIDTH_PX, input)));
    return `${px}px`;
  }

  if (typeof input === 'string') {
    const value = input.trim();
    if (!value) {
      return DEFAULT_CARD_WIDTH;
    }

    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > 0) {
      const px = Math.round(Math.max(MIN_CARD_WIDTH_PX, Math.min(MAX_CARD_WIDTH_PX, numeric)));
      return `${px}px`;
    }

    return value;
  }

  return DEFAULT_CARD_WIDTH;
}

export function normalizeDurationMs(input: unknown): number {
  const value = Number(input);
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  return Math.min(MAX_DURATION_MS, Math.round(value));
}
