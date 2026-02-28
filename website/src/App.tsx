import { startTransition, useDeferredValue, useState } from "react";

type Section =
  | "Dashboard"
  | "Memory"
  | "Notes"
  | "Nutrition"
  | "Apps"
  | "Preferences"
  | "System";

type Transcript = {
  id: string;
  at: string;
  speaker: string;
  text: string;
  confidence: number;
  clipId?: string;
};

type Note = {
  id: string;
  title: string;
  summary: string;
  at: string;
  relatedTranscriptIds: string[];
};

type Clip = {
  id: string;
  title: string;
  at: string;
  duration: string;
  description: string;
};

type AppIntegration = {
  id: string;
  name: string;
  type: string;
  enabled: boolean;
  address: string;
  schema: string;
  status: "Healthy" | "Degraded";
  description: string;
};

type FoodEntry = {
  id: string;
  product: string;
  brand: string;
  calories: number;
  protein: number;
  fat: number;
  carbs: number;
  serving: string;
  at: string;
};

type Preferences = {
  colorBlindness: string;
  contrast: "Balanced" | "High";
  textScale: "Compact" | "Comfortable" | "Large";
  density: "Dense" | "Relaxed";
  verbosity: "Quiet" | "Balanced" | "Verbose";
  defaultRange: string;
};

const sections: Section[] = [
  "Dashboard",
  "Memory",
  "Notes",
  "Nutrition",
  "Apps",
  "Preferences",
  "System",
];

const transcripts: Transcript[] = [
  {
    id: "t-1042",
    at: "2026-02-28T08:10:00",
    speaker: "User",
    text: "Summarize the lunch conversation and tag the camera clip.",
    confidence: 0.98,
    clipId: "clip-77",
  },
  {
    id: "t-1041",
    at: "2026-02-28T07:54:00",
    speaker: "Cleo",
    text: "I found two notes related to your grocery planning from yesterday evening.",
    confidence: 0.96,
  },
  {
    id: "t-1040",
    at: "2026-02-27T20:24:00",
    speaker: "User",
    text: "Remember that the oat bowls from Juniper were surprisingly high in protein.",
    confidence: 0.94,
    clipId: "clip-71",
  },
  {
    id: "t-1039",
    at: "2026-02-27T18:06:00",
    speaker: "User",
    text: "Open the note about the demo rehearsal and show the clip around the first applause.",
    confidence: 0.92,
    clipId: "clip-69",
  },
];

const notes: Note[] = [
  {
    id: "n-22",
    title: "Lunch Debrief",
    summary:
      "Cleo captured a discussion about meal prep, protein targets, and a reminder to compare two snack brands before reordering.",
    at: "2026-02-28T08:12:00",
    relatedTranscriptIds: ["t-1042", "t-1041"],
  },
  {
    id: "n-21",
    title: "Demo Rehearsal",
    summary:
      "The rehearsal note highlights a strong transition after the product intro, a request to shorten the closing, and the point where audience applause started.",
    at: "2026-02-27T18:11:00",
    relatedTranscriptIds: ["t-1039"],
  },
];

const clips: Clip[] = [
  {
    id: "clip-77",
    title: "Kitchen Counter",
    at: "2026-02-28T08:09:00",
    duration: "00:38",
    description: "Short clip linked to the lunch summary request and nutrition capture.",
  },
  {
    id: "clip-71",
    title: "Cafe Table",
    at: "2026-02-27T20:23:00",
    duration: "01:12",
    description: "Meal discussion with visible packaging for the oat bowls.",
  },
  {
    id: "clip-69",
    title: "Demo Room",
    at: "2026-02-27T18:05:00",
    duration: "02:04",
    description: "Practice session with the applause moment near the end of the clip.",
  },
];

const integrationSeed: AppIntegration[] = [
  {
    id: "app-1",
    name: "Calendar Bridge",
    type: "Productivity",
    enabled: true,
    address: "grpc://127.0.0.1:51001",
    schema: "calendar.lookup.v1",
    status: "Healthy",
    description: "Reads agenda context so Cleo can anchor notes to upcoming events.",
  },
  {
    id: "app-2",
    name: "Macro Importer",
    type: "Nutrition",
    enabled: true,
    address: "grpc://127.0.0.1:51009",
    schema: "nutrition.macros.v1",
    status: "Healthy",
    description: "Stores structured food macro records for later review.",
  },
  {
    id: "app-3",
    name: "Home Devices",
    type: "Automation",
    enabled: false,
    address: "grpc://127.0.0.1:51014",
    schema: "iot.scene.v1",
    status: "Degraded",
    description: "Lets Cleo inspect scenes and recent device actions.",
  },
];

const foodEntries: FoodEntry[] = [
  {
    id: "f-1",
    product: "High-Protein Oat Bowl",
    brand: "Juniper",
    calories: 330,
    protein: 24,
    fat: 8,
    carbs: 41,
    serving: "1 bowl",
    at: "2026-02-27T20:25:00",
  },
  {
    id: "f-2",
    product: "Greek Yogurt",
    brand: "North Peak",
    calories: 140,
    protein: 17,
    fat: 2,
    carbs: 9,
    serving: "170 g",
    at: "2026-02-28T08:08:00",
  },
];

const activity = [
  { kind: "Transcript", label: "Wake phrase triggered follow-up summary", at: "2m ago" },
  { kind: "Note", label: "Lunch Debrief note saved", at: "4m ago" },
  { kind: "Clip", label: "Kitchen Counter clip indexed", at: "7m ago" },
  { kind: "App", label: "Macro Importer sync completed", at: "11m ago" },
];

const defaultPrefs: Preferences = {
  colorBlindness: "None",
  contrast: "Balanced",
  textScale: "Comfortable",
  density: "Relaxed",
  verbosity: "Balanced",
  defaultRange: "24 hours",
};

function App() {
  const [section, setSection] = useState<Section>("Dashboard");
  const [search, setSearch] = useState("");
  const [range, setRange] = useState("7 days");
  const [selectedTranscriptId, setSelectedTranscriptId] = useState(transcripts[0].id);
  const [selectedNoteId, setSelectedNoteId] = useState(notes[0].id);
  const [apps, setApps] = useState<AppIntegration[]>(integrationSeed);
  const [selectedAppId, setSelectedAppId] = useState(integrationSeed[0].id);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [appFilter, setAppFilter] = useState("All");
  const [preferences, setPreferences] = useState<Preferences>(defaultPrefs);
  const deferredSearch = useDeferredValue(search);

  const filteredTranscripts = transcripts.filter((entry) => {
    const matchesQuery =
      entry.text.toLowerCase().includes(deferredSearch.toLowerCase()) ||
      entry.id.toLowerCase().includes(deferredSearch.toLowerCase());
    return matchesQuery && inRange(entry.at, range);
  });

  const selectedTranscript =
    filteredTranscripts.find((entry) => entry.id === selectedTranscriptId) ?? filteredTranscripts[0] ?? null;
  const selectedNote = notes.find((entry) => entry.id === selectedNoteId) ?? notes[0];
  const selectedApp = apps.find((entry) => entry.id === selectedAppId) ?? apps[0];
  const selectedClip =
    clips.find((entry) => entry.id === selectedClipId) ?? null;

  const visibleApps = apps.filter((entry) => {
    if (appFilter === "Enabled") {
      return entry.enabled;
    }
    if (appFilter === "Disabled") {
      return !entry.enabled;
    }
    return true;
  });
  const enabledAppCount = apps.filter((entry) => entry.enabled).length;

  const densityClass = preferences.density === "Dense" ? "density-dense" : "density-relaxed";
  const contrastClass = preferences.contrast === "High" ? "contrast-high" : "contrast-balanced";
  const scaleClass =
    preferences.textScale === "Compact"
      ? "scale-compact"
      : preferences.textScale === "Large"
        ? "scale-large"
        : "scale-comfortable";

  return (
    <div className={`app-shell ${densityClass} ${contrastClass} ${scaleClass}`}>
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">Cleo</p>
          <h1>Control Center</h1>
          <p className="muted">Personal memory dashboard for notes, history, nutrition, and system controls.</p>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {sections.map((item) => (
            <button
              key={item}
              className={item === section ? "nav-item active" : "nav-item"}
              onClick={() => {
                startTransition(() => setSection(item));
              }}
              type="button"
            >
              <span>{item}</span>
              <small>{sectionHint(item)}</small>
            </button>
          ))}
        </nav>
        <div className="status-card">
          <p className="eyebrow">System Health</p>
          <div className="health-row">
            <span>Sensor Service</span>
            <strong>Reachable</strong>
          </div>
          <div className="health-row">
            <span>Data Service</span>
            <strong>Reachable</strong>
          </div>
          <div className="health-row">
            <span>Assistant Service</span>
            <strong className="warn">Warm start</strong>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Workspace</p>
            <h2>{section}</h2>
          </div>
          <div className="top-actions">
            <button className="ghost-button" type="button" onClick={() => startTransition(() => setSection("Memory"))}>
              Search Memory
            </button>
            <button className="ghost-button" type="button" onClick={() => startTransition(() => setSection("Notes"))}>
              Review Notes
            </button>
            <button
              className="solid-button"
              type="button"
              onClick={() => startTransition(() => setSection("Preferences"))}
            >
              Edit Preferences
            </button>
          </div>
        </header>

        {section === "Dashboard" && (
          <section className="stacked-layout">
            <div className="stats-grid">
              <StatCard label="Recent transcriptions" value="148" meta="12 in the last 24 hours" />
              <StatCard label="Saved notes" value="26" meta="2 created this morning" />
              <StatCard label="Indexed clips" value="63" meta="7 linked to conversations" />
              <StatCard
                label="Enabled apps"
                value={`${enabledAppCount} / ${apps.length}`}
                meta="One integration currently needs attention"
              />
            </div>

            <div className="hero-grid">
              <section className="surface-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Recent Activity</p>
                    <h3>What Cleo captured recently</h3>
                  </div>
                </div>
                <div className="activity-list">
                  {activity.map((entry) => (
                    <div className="activity-row" key={`${entry.kind}-${entry.label}`}>
                      <span className="pill">{entry.kind}</span>
                      <div>
                        <strong>{entry.label}</strong>
                        <p className="muted">{entry.at}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="surface-panel accent-panel">
                <p className="eyebrow">Quick Summary</p>
                <h3>Today feels organized</h3>
                <p>
                  Cleo indexed a new meal entry, generated a lunch note, and attached the latest kitchen clip to the
                  same memory window. Search is up to date and ready for follow-up review.
                </p>
                <div className="quick-link-row">
                  <button className="ghost-button" type="button" onClick={() => setSection("Memory")}>
                    Open Memory
                  </button>
                  <button className="ghost-button" type="button" onClick={() => setSection("Nutrition")}>
                    View Nutrition
                  </button>
                </div>
              </section>
            </div>
          </section>
        )}

        {section === "Memory" && (
          <section className="stacked-layout">
            <section className="surface-panel">
              <div className="toolbar">
                <label className="field">
                  <span>Global search</span>
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search transcripts, ids, and remembered moments"
                  />
                </label>
                <label className="field small-field">
                  <span>Time range</span>
                  <select value={range} onChange={(event) => setRange(event.target.value)}>
                    <option>24 hours</option>
                    <option>7 days</option>
                    <option>30 days</option>
                  </select>
                </label>
              </div>

              <div className="memory-grid">
                <div className="table-shell">
                  <div className="table-head">
                    <span>Timestamp</span>
                    <span>Speaker</span>
                    <span>Transcript</span>
                    <span>Confidence</span>
                  </div>
                  {filteredTranscripts.length === 0 && (
                    <div className="empty-state">
                      <h4>No matching memories</h4>
                      <p>Try a broader range or remove part of the query.</p>
                    </div>
                  )}
                  {filteredTranscripts.map((entry) => (
                    <button
                      key={entry.id}
                      className={entry.id === selectedTranscript?.id ? "table-row active" : "table-row"}
                      type="button"
                      onClick={() => setSelectedTranscriptId(entry.id)}
                    >
                      <span>{formatDate(entry.at)}</span>
                      <span>{entry.speaker}</span>
                      <span>{entry.text}</span>
                      <span>{Math.round(entry.confidence * 100)}%</span>
                    </button>
                  ))}
                </div>

                <aside className="detail-panel">
                  <p className="eyebrow">Detail</p>
                  {selectedTranscript ? (
                    <>
                      <h3>{selectedTranscript.id}</h3>
                      <p className="muted">{formatDate(selectedTranscript.at)}</p>
                      <p>{selectedTranscript.text}</p>
                      <div className="meta-grid">
                        <div>
                          <span className="meta-label">Confidence</span>
                          <strong>{Math.round(selectedTranscript.confidence * 100)}%</strong>
                        </div>
                        <div>
                          <span className="meta-label">Adjacent entries</span>
                          <strong>3</strong>
                        </div>
                      </div>
                      <button
                        className="solid-button"
                        type="button"
                        disabled={!selectedTranscript.clipId}
                        onClick={() => setSelectedClipId(selectedTranscript.clipId ?? null)}
                      >
                        {selectedTranscript.clipId ? "Open related clip" : "No clip attached"}
                      </button>
                    </>
                  ) : (
                    <p className="muted">No transcript selected.</p>
                  )}
                </aside>
              </div>
            </section>
          </section>
        )}

        {section === "Notes" && (
          <section className="notes-grid">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Note Summaries</p>
                  <h3>Generated reading view</h3>
                </div>
              </div>
              <div className="note-list">
                {notes.map((entry) => (
                  <button
                    key={entry.id}
                    className={entry.id === selectedNote.id ? "note-card active" : "note-card"}
                    type="button"
                    onClick={() => setSelectedNoteId(entry.id)}
                  >
                    <p className="eyebrow">{formatDate(entry.at)}</p>
                    <h4>{entry.title}</h4>
                    <p>{entry.summary}</p>
                  </button>
                ))}
              </div>
            </section>

            <aside className="surface-panel note-detail">
              <p className="eyebrow">Detail</p>
              <h3>{selectedNote.title}</h3>
              <p className="muted">{formatDate(selectedNote.at)}</p>
              <p>{selectedNote.summary}</p>
              <div className="linked-actions">
                <button className="ghost-button" type="button" onClick={() => setSection("Memory")}>
                  Jump to transcriptions
                </button>
                <button className="ghost-button" type="button" onClick={() => setSelectedClipId("clip-69")}>
                  Open related clip
                </button>
              </div>
            </aside>
          </section>
        )}

        {section === "Nutrition" && (
          <section className="stacked-layout">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Food Macros</p>
                  <h3>Captured nutrition entries</h3>
                </div>
                <span className="notice-badge">Backend read RPC still required for production</span>
              </div>
              <div className="table-shell">
                <div className="table-head nutrition-head">
                  <span>Product</span>
                  <span>Brand</span>
                  <span>Calories</span>
                  <span>Protein</span>
                  <span>Fat</span>
                  <span>Carbs</span>
                  <span>Serving</span>
                  <span>Recorded</span>
                </div>
                {foodEntries.map((entry) => (
                  <div className="table-row nutrition-row" key={entry.id}>
                    <span>{entry.product}</span>
                    <span>{entry.brand}</span>
                    <span>{entry.calories}</span>
                    <span>{entry.protein} g</span>
                    <span>{entry.fat} g</span>
                    <span>{entry.carbs} g</span>
                    <span>{entry.serving}</span>
                    <span>{formatDate(entry.at)}</span>
                  </div>
                ))}
              </div>
            </section>
          </section>
        )}

        {section === "Apps" && (
          <section className="apps-grid">
            <section className="surface-panel">
              <div className="toolbar">
                <label className="field small-field">
                  <span>Filter</span>
                  <select value={appFilter} onChange={(event) => setAppFilter(event.target.value)}>
                    <option>All</option>
                    <option>Enabled</option>
                    <option>Disabled</option>
                  </select>
                </label>
              </div>
              <div className="app-list">
                {visibleApps.map((entry) => (
                  <button
                    key={entry.id}
                    className={entry.id === selectedApp?.id ? "app-card active" : "app-card"}
                    type="button"
                    onClick={() => setSelectedAppId(entry.id)}
                  >
                    <div className="app-card-top">
                      <h4>{entry.name}</h4>
                      <span className={entry.enabled ? "pill success" : "pill muted-pill"}>
                        {entry.enabled ? "Enabled" : "Disabled"}
                      </span>
                    </div>
                    <p>{entry.description}</p>
                    <small>{entry.type}</small>
                  </button>
                ))}
                {visibleApps.length === 0 && (
                  <div className="empty-state">
                    <h4>No apps match this filter</h4>
                    <p>Switch the filter to inspect all registered integrations.</p>
                  </div>
                )}
              </div>
            </section>

            <aside className="surface-panel detail-panel">
              <p className="eyebrow">Integration Detail</p>
                      <h3>{selectedApp?.name ?? "No app selected"}</h3>
                      <div className="meta-stack">
                        <div className="kv-row">
                          <span>Status</span>
                          <strong>{selectedApp?.status ?? "Unavailable"}</strong>
                        </div>
                        <div className="kv-row">
                          <span>Address</span>
                          <strong>{selectedApp?.address ?? "Unavailable"}</strong>
                        </div>
                        <div className="kv-row">
                          <span>Schema</span>
                          <strong>{selectedApp?.schema ?? "Unavailable"}</strong>
                        </div>
                      </div>
                      <button
                className={selectedApp?.enabled ? "ghost-button" : "solid-button"}
                        type="button"
                        disabled={!selectedApp}
                        onClick={() => {
                          if (!selectedApp) {
                            return;
                          }
                          setApps((current) =>
                            current.map((entry) =>
                              entry.id === selectedApp.id ? { ...entry, enabled: !entry.enabled } : entry,
                            ),
                          );
                          setAppFilter("All");
                        }}
                      >
                        {selectedApp?.enabled ? "Disable app" : "Enable app"}
                      </button>
            </aside>
          </section>
        )}

        {section === "Preferences" && (
          <section className="stacked-layout">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Persistent Preferences</p>
                  <h3>Structured client-side schema</h3>
                </div>
              </div>

              <div className="preferences-grid">
                <PreferenceSelect
                  label="Color blindness type"
                  value={preferences.colorBlindness}
                  options={["None", "Protanopia", "Deuteranopia", "Tritanopia"]}
                  onChange={(value) => setPreferences({ ...preferences, colorBlindness: value })}
                />
                <PreferenceSelect
                  label="Preferred contrast mode"
                  value={preferences.contrast}
                  options={["Balanced", "High"]}
                  onChange={(value) =>
                    setPreferences({ ...preferences, contrast: value as Preferences["contrast"] })
                  }
                />
                <PreferenceSelect
                  label="Text size"
                  value={preferences.textScale}
                  options={["Compact", "Comfortable", "Large"]}
                  onChange={(value) =>
                    setPreferences({ ...preferences, textScale: value as Preferences["textScale"] })
                  }
                />
                <PreferenceSelect
                  label="UI density"
                  value={preferences.density}
                  options={["Dense", "Relaxed"]}
                  onChange={(value) =>
                    setPreferences({ ...preferences, density: value as Preferences["density"] })
                  }
                />
                <PreferenceSelect
                  label="Notification verbosity"
                  value={preferences.verbosity}
                  options={["Quiet", "Balanced", "Verbose"]}
                  onChange={(value) =>
                    setPreferences({ ...preferences, verbosity: value as Preferences["verbosity"] })
                  }
                />
                <PreferenceSelect
                  label="Default time range"
                  value={preferences.defaultRange}
                  options={["24 hours", "7 days", "30 days"]}
                  onChange={(value) => setPreferences({ ...preferences, defaultRange: value })}
                />
              </div>
            </section>
          </section>
        )}

        {section === "System" && (
          <section className="stacked-layout">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Operational Visibility</p>
                  <h3>Lightweight diagnostics</h3>
                </div>
              </div>
              <div className="stats-grid">
                <StatCard label="Backend version" value="0.1.0" meta="services/main.py" />
                <StatCard label="Last sync" value="08:12" meta="Data service index refresh" />
                <StatCard label="Queue depth" value="3" meta="Pending background tasks" />
                <StatCard label="Memory store" value="Healthy" meta="FAISS + SQLite online" />
              </div>
            </section>
          </section>
        )}
      </main>

      {selectedClip && (
        <div className="modal-backdrop" role="presentation" onClick={() => setSelectedClipId(null)}>
          <section className="modal-card" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Video Clip</p>
                <h3>{selectedClip.title}</h3>
              </div>
              <button className="ghost-button" type="button" onClick={() => setSelectedClipId(null)}>
                Close
              </button>
            </div>
            <div className="clip-preview">
              <div className="clip-frame">
                <span>Inline preview placeholder</span>
              </div>
              <div className="meta-stack">
                <div className="kv-row">
                  <span>Recorded</span>
                  <strong>{formatDate(selectedClip.at)}</strong>
                </div>
                <div className="kv-row">
                  <span>Duration</span>
                  <strong>{selectedClip.duration}</strong>
                </div>
              </div>
              <p>{selectedClip.description}</p>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function PreferenceSelect(props: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="field preference-field">
      <span>{props.label}</span>
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function StatCard(props: { label: string; value: string; meta: string }) {
  return (
    <section className="stat-card">
      <p>{props.label}</p>
      <h3>{props.value}</h3>
      <small>{props.meta}</small>
    </section>
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function inRange(value: string, range: string) {
  const now = new Date("2026-02-28T12:00:00");
  const target = new Date(value);
  const diff = now.getTime() - target.getTime();
  const day = 24 * 60 * 60 * 1000;
  if (range === "24 hours") {
    return diff <= day;
  }
  if (range === "7 days") {
    return diff <= 7 * day;
  }
  return diff <= 30 * day;
}

function sectionHint(section: Section) {
  switch (section) {
    case "Dashboard":
      return "Recent activity";
    case "Memory":
      return "Search + clips";
    case "Notes":
      return "Summaries";
    case "Nutrition":
      return "Food macros";
    case "Apps":
      return "Integrations";
    case "Preferences":
      return "Accessibility";
    case "System":
      return "Diagnostics";
  }
}

export default App;
