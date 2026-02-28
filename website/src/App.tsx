import { startTransition, useEffect, useState, type FormEvent } from "react";

type Section =
  | "Dashboard"
  | "Memory"
  | "Faces"
  | "Notes"
  | "Nutrition"
  | "Apps"
  | "Preferences"
  | "System";

type FoodEntry = {
  id: number;
  productName: string;
  brand: string;
  caloriesKcal: number | null;
  proteinG: number | null;
  fatG: number | null;
  carbsG: number | null;
  servingSize: string;
  servingQuantity: number | null;
  basis: string;
  recordedAt: number;
};

type NoteEntry = {
  id: number;
  summaryText: string;
  startTimestamp: number;
  endTimestamp: number;
  createdAt: number;
};

type MemoryClipEntry = {
  clipId: number;
  score: number;
  startTimestamp: number;
  endTimestamp: number;
  numFrames: number;
  videoUrl: string;
};

type FaceEntry = {
  faceId: number;
  name: string;
  firstSeen: number;
  lastSeen: number;
  seenCount: number;
  confidence: number | null;
  imageUrl: string;
  collageImageUrls: string[];
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
  "Faces",
  "Notes",
  "Nutrition",
  "Apps",
  "Preferences",
  "System",
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
  const [foodEntries, setFoodEntries] = useState<FoodEntry[]>([]);
  const [foodError, setFoodError] = useState<string | null>(null);
  const [foodLoading, setFoodLoading] = useState(true);
  const [noteEntries, setNoteEntries] = useState<NoteEntry[]>([]);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [noteLoading, setNoteLoading] = useState(true);
  const [selectedNoteId, setSelectedNoteId] = useState<number | null>(null);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryStartTime, setMemoryStartTime] = useState("");
  const [memoryEndTime, setMemoryEndTime] = useState("");
  const [memoryResults, setMemoryResults] = useState<MemoryClipEntry[]>([]);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [memoryHasSearched, setMemoryHasSearched] = useState(false);
  const [selectedMemoryClipId, setSelectedMemoryClipId] = useState<number | null>(null);
  const [faceEntries, setFaceEntries] = useState<FaceEntry[]>([]);
  const [faceError, setFaceError] = useState<string | null>(null);
  const [faceLoading, setFaceLoading] = useState(true);
  const [faceSavingId, setFaceSavingId] = useState<number | null>(null);
  const [faceSaveError, setFaceSaveError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<Preferences>(defaultPrefs);

  const densityClass = preferences.density === "Dense" ? "density-dense" : "density-relaxed";
  const contrastClass = preferences.contrast === "High" ? "contrast-high" : "contrast-balanced";
  const scaleClass =
    preferences.textScale === "Compact"
      ? "scale-compact"
      : preferences.textScale === "Large"
        ? "scale-large"
        : "scale-comfortable";
  const websiteApiState =
    foodLoading || noteLoading || faceLoading
      ? "Checking"
      : foodError || noteError || faceError
        ? "Unavailable"
        : "Ready";
  const memorySummary =
    memoryLoading ? "Searching" : memoryError ? "Error" : memoryHasSearched ? String(memoryResults.length) : "Ready";
  const nutritionSummary =
    foodLoading
      ? "Refreshing"
      : foodError
        ? "Unavailable"
        : String(foodEntries.length);
  const notesSummary =
    noteLoading
      ? "Refreshing"
      : noteError
        ? "Unavailable"
        : String(noteEntries.length);
  const facesSummary =
    faceLoading
      ? "Refreshing"
      : faceError
        ? "Unavailable"
        : String(faceEntries.length);
  const memorySummaryMeta = memoryError
    ? memoryError
    : memoryHasSearched
      ? `Latest search returned ${memoryResults.length} clip${memoryResults.length === 1 ? "" : "s"}`
      : "Vector search is connected";
  const selectedNote = noteEntries.find((entry) => entry.id === selectedNoteId) ?? noteEntries[0] ?? null;
  const selectedMemoryClip =
    memoryResults.find((entry) => entry.clipId === selectedMemoryClipId) ?? memoryResults[0] ?? null;

  useEffect(() => {
    const controller = new AbortController();

    async function loadFoodMacros() {
      try {
        setFoodLoading(true);
        setFoodError(null);

        const response = await fetch("/api/food-macros?limit=100", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const payload = (await response.json()) as {
          entries?: Array<{
            id: number;
            productName: string;
            brand: string;
            caloriesKcal: number | null;
            proteinG: number | null;
            fatG: number | null;
            carbsG: number | null;
            servingSize: string;
            servingQuantity: number | null;
            basis: string;
            recordedAt: number;
          }>;
        };
        setFoodEntries(payload.entries ?? []);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setFoodError(error instanceof Error ? error.message : "Unable to load nutrition entries.");
        setFoodEntries([]);
      } finally {
        if (!controller.signal.aborted) {
          setFoodLoading(false);
        }
      }
    }

    void loadFoodMacros();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    async function loadFaces() {
      try {
        setFaceLoading(true);
        setFaceError(null);

        const response = await fetch("/api/faces?limit=200", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const payload = (await response.json()) as {
          entries?: FaceEntry[];
        };
        setFaceEntries(payload.entries ?? []);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setFaceError(error instanceof Error ? error.message : "Unable to load face groups.");
        setFaceEntries([]);
      } finally {
        if (!controller.signal.aborted) {
          setFaceLoading(false);
        }
      }
    }

    void loadFaces();

    return () => controller.abort();
  }, []);

  async function searchMemory(queryOverride?: string) {
    const query = (queryOverride ?? memoryQuery).trim();
    setMemoryHasSearched(true);
    const startTimestamp = parseLocalDateTimeToUnix(memoryStartTime);
    const endTimestamp = parseLocalDateTimeToUnix(memoryEndTime);
    if (!query && (startTimestamp === null || endTimestamp === null)) {
      setMemoryError("Leave the text blank only when both start and end times are set.");
      setMemoryResults([]);
      setSelectedMemoryClipId(null);
      return;
    }
    if (
      startTimestamp !== null &&
      endTimestamp !== null &&
      startTimestamp > endTimestamp
    ) {
      setMemoryError("Start time must be before end time.");
      setMemoryResults([]);
      setSelectedMemoryClipId(null);
      return;
    }

    try {
      setMemoryLoading(true);
      setMemoryError(null);

      const searchParams = new URLSearchParams({
        q: query,
        limit: "8",
      });
      if (startTimestamp !== null) {
        searchParams.set("startTimestamp", String(startTimestamp));
      }
      if (endTimestamp !== null) {
        searchParams.set("endTimestamp", String(endTimestamp));
      }

      const response = await fetch(`/api/search?${searchParams.toString()}`);
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { message?: string } | null;
        throw new Error(payload?.message ?? `Request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as {
        entries?: MemoryClipEntry[];
      };
      const entries = payload.entries ?? [];
      setMemoryResults(entries);
      setSelectedMemoryClipId((current) => {
        if (entries.length === 0) {
          return null;
        }
        if (current !== null && entries.some((entry) => entry.clipId === current)) {
          return current;
        }
        return entries[0].clipId;
      });
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : "Unable to search saved video clips.");
      setMemoryResults([]);
      setSelectedMemoryClipId(null);
    } finally {
      setMemoryLoading(false);
    }
  }

  function handleMemorySearchSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void searchMemory();
  }

  function handleFaceNameChange(faceId: number, value: string) {
    setFaceEntries((current) =>
      current.map((entry) =>
        entry.faceId === faceId
          ? {
              ...entry,
              name: value,
            }
          : entry,
      ),
    );
  }

  async function saveFaceName(faceId: number) {
    const face = faceEntries.find((entry) => entry.faceId === faceId);
    if (!face) {
      return;
    }

    try {
      setFaceSavingId(faceId);
      setFaceSaveError(null);

      const response = await fetch(`/api/faces/${faceId}/name`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: face.name }),
      });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { message?: string } | null;
        throw new Error(payload?.message ?? `Request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as {
        name?: string;
      };
      setFaceEntries((current) =>
        current.map((entry) =>
          entry.faceId === faceId
            ? {
                ...entry,
                name: payload.name ?? entry.name,
              }
            : entry,
        ),
      );
    } catch (error) {
      setFaceSaveError(error instanceof Error ? error.message : "Unable to save the face name.");
    } finally {
      setFaceSavingId((current) => (current === faceId ? null : current));
    }
  }

  useEffect(() => {
    const controller = new AbortController();

    async function loadNotes() {
      try {
        setNoteLoading(true);
        setNoteError(null);

        const response = await fetch("/api/notes?limit=100", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const payload = (await response.json()) as {
          entries?: Array<{
            id: number;
            summaryText: string;
            startTimestamp: number;
            endTimestamp: number;
            createdAt: number;
          }>;
        };
        const entries = payload.entries ?? [];
        setNoteEntries(entries);
        setSelectedNoteId((current) => {
          if (entries.length === 0) {
            return null;
          }
          if (current !== null && entries.some((entry) => entry.id === current)) {
            return current;
          }
          return entries[0].id;
        });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }
        setNoteError(error instanceof Error ? error.message : "Unable to load note summaries.");
        setNoteEntries([]);
        setSelectedNoteId(null);
      } finally {
        if (!controller.signal.aborted) {
          setNoteLoading(false);
        }
      }
    }

    void loadNotes();

    return () => controller.abort();
  }, []);

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
            <span>Website API</span>
            <strong>{websiteApiState}</strong>
          </div>
          <div className="health-row">
            <span>Nutrition Feed</span>
            <strong>{foodLoading ? "Loading" : foodError ? "Error" : "Synced"}</strong>
          </div>
          <div className="health-row">
            <span>Face Review</span>
            <strong>{faceLoading ? "Loading" : faceError ? "Error" : "Synced"}</strong>
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
            <button className="ghost-button" type="button" onClick={() => startTransition(() => setSection("Faces"))}>
              Review Faces
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
              <StatCard label="Nutrition entries" value={nutritionSummary} meta="Live count from local database" />
              <StatCard label="Memory" value={memorySummary} meta={memorySummaryMeta} />
              <StatCard label="Notes" value={notesSummary} meta={noteError ?? "Live count from local database"} />
              <StatCard label="Faces" value={facesSummary} meta={faceError ?? "Named face groups from the local database"} />
            </div>

            <div className="hero-grid">
              <section className="surface-panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Live Data</p>
                    <h3>Current website wiring</h3>
                  </div>
                </div>
                <div className="empty-state">
                  <h4>Notes, nutrition, memory search, and faces are live</h4>
                  <p>
                    The website no longer renders fake records. Notes, nutrition, and face groups read from the
                    backend, and memory search queries saved video embeddings through the website API.
                  </p>
                </div>
              </section>

              <section className="surface-panel accent-panel">
                <p className="eyebrow">Next Up</p>
                <h3>Connect the remaining sections</h3>
                <p>
                  Apps still need live backend routes. The other sections on this dashboard now render persisted data.
                </p>
                <div className="quick-link-row">
                  <button className="ghost-button" type="button" onClick={() => setSection("Faces")}>
                    View Faces
                  </button>
                  <button className="ghost-button" type="button" onClick={() => setSection("Memory")}>
                    View Memory
                  </button>
                </div>
              </section>
            </div>
          </section>
        )}

        {section === "Memory" && (
          <section className="memory-grid">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Video Memory Search</p>
                  <h3>Find saved clips by text</h3>
                </div>
                <span className="pill success">Vector DB</span>
              </div>
              <form className="toolbar" onSubmit={handleMemorySearchSubmit}>
                <label className="field">
                  <span>Search prompt</span>
                  <input
                    type="search"
                    value={memoryQuery}
                    placeholder="What happened near the whiteboard?"
                    onChange={(event) => setMemoryQuery(event.target.value)}
                  />
                </label>
                <label className="field small-field">
                  <span>Start time</span>
                  <input
                    type="datetime-local"
                    value={memoryStartTime}
                    onChange={(event) => setMemoryStartTime(event.target.value)}
                  />
                </label>
                <label className="field small-field">
                  <span>End time</span>
                  <input
                    type="datetime-local"
                    value={memoryEndTime}
                    onChange={(event) => setMemoryEndTime(event.target.value)}
                  />
                </label>
                <button className="solid-button" type="submit" disabled={memoryLoading}>
                  {memoryLoading ? "Searching..." : "Search clips"}
                </button>
              </form>
              <div className="table-shell">
                <div className="table-head memory-head">
                  <span>Clip</span>
                  <span>Time window</span>
                  <span>Match</span>
                  <span>Frames</span>
                </div>
                {!memoryHasSearched && (
                  <div className="empty-state">
                    <h4>Search your saved clips</h4>
                    <p>Enter a description for vector search, or leave it blank and supply both timestamps to browse clips in that window.</p>
                  </div>
                )}
                {memoryHasSearched && memoryLoading && (
                  <div className="empty-state">
                    <h4>Searching video memory</h4>
                    <p>Embedding your text prompt and looking up the nearest saved clips.</p>
                  </div>
                )}
                {memoryHasSearched && !memoryLoading && memoryError && (
                  <div className="empty-state">
                    <h4>Memory search unavailable</h4>
                    <p>{memoryError}</p>
                  </div>
                )}
                {memoryHasSearched && !memoryLoading && !memoryError && memoryResults.length === 0 && (
                  <div className="empty-state">
                    <h4>No matching clips</h4>
                    <p>Try a broader description or search for a different moment.</p>
                  </div>
                )}
                {memoryHasSearched &&
                  !memoryLoading &&
                  !memoryError &&
                  memoryResults.map((entry) => (
                    <button
                      key={entry.clipId}
                      className={entry.clipId === selectedMemoryClip?.clipId ? "table-row memory-row active" : "table-row memory-row"}
                      type="button"
                      onClick={() => setSelectedMemoryClipId(entry.clipId)}
                    >
                      <span>Clip #{entry.clipId}</span>
                      <span>
                        {formatUnixTimestamp(entry.startTimestamp)} to {formatUnixTimestamp(entry.endTimestamp)}
                      </span>
                      <span>{formatScore(entry.score)}</span>
                      <span>{entry.numFrames}</span>
                    </button>
                  ))}
              </div>
            </section>

            <aside className="surface-panel detail-panel">
              <p className="eyebrow">Clip Preview</p>
              {!selectedMemoryClip && (
                <div className="empty-state">
                  <h4>No clip selected</h4>
                  <p>Run a search and select a result to play it here.</p>
                </div>
              )}
              {selectedMemoryClip && (
                <div className="clip-preview">
                  <div className="meta-grid">
                    <div>
                      <span className="meta-label">Clip</span>
                      <strong>#{selectedMemoryClip.clipId}</strong>
                    </div>
                    <div>
                      <span className="meta-label">Similarity</span>
                      <strong>{formatScore(selectedMemoryClip.score)}</strong>
                    </div>
                  </div>
                  <div className="meta-grid">
                    <div>
                      <span className="meta-label">Started</span>
                      <strong>{formatUnixTimestamp(selectedMemoryClip.startTimestamp)}</strong>
                    </div>
                    <div>
                      <span className="meta-label">Duration</span>
                      <strong>{formatDuration(selectedMemoryClip.startTimestamp, selectedMemoryClip.endTimestamp)}</strong>
                    </div>
                  </div>
                  <video
                    className="clip-player"
                    controls
                    preload="metadata"
                    src={selectedMemoryClip.videoUrl}
                  >
                    Your browser does not support MP4 playback.
                  </video>
                </div>
              )}
            </aside>
          </section>
        )}

        {section === "Faces" && (
          <section className="stacked-layout">
            <section className="surface-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Face Groups</p>
                  <h3>Review tracked identities</h3>
                </div>
                <span className="pill success">{faceEntries.length} saved</span>
              </div>
              {faceSaveError && (
                <div className="inline-alert">
                  <strong>Save failed.</strong> {faceSaveError}
                </div>
              )}
              <div className="face-grid">
                {faceLoading && (
                  <div className="empty-state face-empty-state">
                    <h4>Loading face groups</h4>
                    <p>Fetching representative images and identity metadata from the local database.</p>
                  </div>
                )}
                {!faceLoading && faceError && (
                  <div className="empty-state face-empty-state">
                    <h4>Face groups unavailable</h4>
                    <p>{faceError}</p>
                  </div>
                )}
                {!faceLoading && !faceError && faceEntries.length === 0 && (
                  <div className="empty-state face-empty-state">
                    <h4>No faces saved yet</h4>
                    <p>Face groups will appear here after face detection stores embeddings.</p>
                  </div>
                )}
                {!faceLoading &&
                  !faceError &&
                  faceEntries.map((entry) => (
                    <article className="face-card" key={entry.faceId}>
                      <div className="face-collage">
                        {(entry.collageImageUrls.length > 0 ? entry.collageImageUrls : [entry.imageUrl]).map((url, index) => (
                          <img
                            className="face-image"
                            key={`${entry.faceId}-${url}-${index}`}
                            src={url}
                            alt={entry.name ? `${entry.name} sighting ${index + 1}` : `Face group ${entry.faceId} sighting ${index + 1}`}
                            loading="lazy"
                          />
                        ))}
                      </div>
                      <div className="panel-heading">
                        <div>
                          <p className="eyebrow">Face #{entry.faceId}</p>
                          <h4>{entry.name.trim() || "Unnamed"}</h4>
                        </div>
                        <span className="pill success">{entry.seenCount} sightings</span>
                      </div>
                      <div className="meta-stack">
                        <div>
                          <span className="meta-label">First seen</span>
                          <strong>{formatUnixTimestamp(entry.firstSeen)}</strong>
                        </div>
                        <div>
                          <span className="meta-label">Last seen</span>
                          <strong>{formatUnixTimestamp(entry.lastSeen)}</strong>
                        </div>
                        <div>
                          <span className="meta-label">Confidence</span>
                          <strong>{entry.confidence === null ? "--" : `${entry.confidence.toFixed(1)}%`}</strong>
                        </div>
                      </div>
                      <div className="face-form-row">
                        <label className="field face-name-field">
                          <span>Name</span>
                          <input
                            type="text"
                            value={entry.name}
                            placeholder="Add a name"
                            onChange={(event) => handleFaceNameChange(entry.faceId, event.target.value)}
                          />
                        </label>
                        <button
                          className="solid-button"
                          type="button"
                          disabled={faceSavingId === entry.faceId}
                          onClick={() => void saveFaceName(entry.faceId)}
                        >
                          {faceSavingId === entry.faceId ? "Saving..." : "Save"}
                        </button>
                      </div>
                    </article>
                  ))}
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
                    <h3>Captured note history</h3>
                  </div>
                </div>
              <div className="note-list">
                {noteLoading && (
                  <div className="empty-state">
                    <h4>Loading note summaries</h4>
                    <p>Reading the latest saved notes from the local database.</p>
                  </div>
                )}
                {!noteLoading && noteError && (
                  <div className="empty-state">
                    <h4>Note summaries unavailable</h4>
                    <p>{noteError}</p>
                  </div>
                )}
                {!noteLoading && !noteError && noteEntries.length === 0 && (
                  <div className="empty-state">
                    <h4>No note summaries saved yet</h4>
                    <p>Start and stop notetaking once and the summary will appear here.</p>
                  </div>
                )}
                {!noteLoading &&
                  !noteError &&
                  noteEntries.map((entry) => (
                    <button
                      key={entry.id}
                      className={entry.id === selectedNote?.id ? "note-card active" : "note-card"}
                      type="button"
                      onClick={() => setSelectedNoteId(entry.id)}
                    >
                      <div className="panel-heading">
                        <p className="eyebrow">Note #{entry.id}</p>
                        <span className="pill success">{formatUnixTimestamp(entry.createdAt)}</span>
                      </div>
                      <h4>{truncateText(entry.summaryText, 120)}</h4>
                      <p className="muted">
                        Session {formatUnixTimestamp(entry.startTimestamp)} to {formatUnixTimestamp(entry.endTimestamp)}
                      </p>
                    </button>
                  ))}
              </div>
            </section>

            <aside className="surface-panel note-detail">
              <p className="eyebrow">Detail</p>
              {!selectedNote && (
                <div className="empty-state">
                  <h4>No note selected</h4>
                  <p>This panel will activate once note data is loaded from the backend.</p>
                </div>
              )}
              {selectedNote && (
                <>
                  <h3>Note #{selectedNote.id}</h3>
                  <div className="meta-grid">
                    <div>
                      <span className="meta-label">Created</span>
                      <strong>{formatUnixTimestamp(selectedNote.createdAt)}</strong>
                    </div>
                    <div>
                      <span className="meta-label">Duration</span>
                      <strong>{formatDuration(selectedNote.startTimestamp, selectedNote.endTimestamp)}</strong>
                    </div>
                  </div>
                  <div className="meta-stack">
                    <div>
                      <span className="meta-label">Started</span>
                      <strong>{formatUnixTimestamp(selectedNote.startTimestamp)}</strong>
                    </div>
                    <div>
                      <span className="meta-label">Ended</span>
                      <strong>{formatUnixTimestamp(selectedNote.endTimestamp)}</strong>
                    </div>
                  </div>
                  <p>{selectedNote.summaryText}</p>
                </>
              )}
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
                {foodLoading && (
                  <div className="empty-state">
                    <h4>Loading nutrition entries</h4>
                    <p>Reading the latest food macro records from the local database.</p>
                  </div>
                )}
                {!foodLoading && foodError && (
                  <div className="empty-state">
                    <h4>Nutrition data unavailable</h4>
                    <p>{foodError}</p>
                  </div>
                )}
                {!foodLoading && !foodError && foodEntries.length === 0 && (
                  <div className="empty-state">
                    <h4>No food macros saved yet</h4>
                    <p>Use the food macro tool once and the entry will appear here.</p>
                  </div>
                )}
                {!foodLoading &&
                  !foodError &&
                  foodEntries.map((entry) => (
                    <div className="table-row nutrition-row" key={entry.id}>
                      <span>{entry.productName}</span>
                      <span>{entry.brand || "Unknown"}</span>
                      <span>{formatNumber(entry.caloriesKcal)}</span>
                      <span>{formatMetric(entry.proteinG, "g")}</span>
                      <span>{formatMetric(entry.fatG, "g")}</span>
                      <span>{formatMetric(entry.carbsG, "g")}</span>
                      <span>{formatServing(entry)}</span>
                      <span>{formatUnixTimestamp(entry.recordedAt)}</span>
                    </div>
                  ))}
              </div>
            </section>
          </section>
        )}

        {section === "Apps" && (
          <section className="apps-grid">
            <section className="surface-panel">
              <div className="empty-state">
                <h4>No app integrations loaded</h4>
                <p>Static app cards were removed. Connect the apps API before enabling this section.</p>
              </div>
            </section>

            <aside className="surface-panel detail-panel">
              <p className="eyebrow">Integration Detail</p>
              <div className="empty-state">
                <h4>No app selected</h4>
                <p>This panel will show live integration metadata after the apps endpoint is wired.</p>
              </div>
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
                  <h3>Connected services</h3>
                </div>
              </div>
              <div className="stats-grid">
                <StatCard label="Website API" value={websiteApiState} meta="Backs the /api routes used by the website" />
                <StatCard
                  label="Nutrition fetch"
                  value={foodLoading ? "Loading" : foodError ? "Error" : "Healthy"}
                  meta={foodError ?? "Derived from the food macros endpoint"}
                />
                <StatCard
                  label="Notes fetch"
                  value={noteLoading ? "Loading" : noteError ? "Error" : "Healthy"}
                  meta={noteError ?? "Derived from the notes endpoint"}
                />
                <StatCard
                  label="Faces fetch"
                  value={faceLoading ? "Loading" : faceError ? "Error" : "Healthy"}
                  meta={faceError ?? "Derived from the face groups endpoint"}
                />
                <StatCard
                  label="Memory search"
                  value={memoryLoading ? "Loading" : memoryError ? "Error" : "Healthy"}
                  meta={memoryError ?? "Derived from the clip search endpoint"}
                />
                <StatCard label="Mock data" value="Removed" meta="This page no longer shows seeded values" />
                <StatCard label="Apps" value="Pending" meta="Apps are still not connected" />
              </div>
            </section>
          </section>
        )}
      </main>
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

function formatUnixTimestamp(value: number) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value * 1000));
}

function parseLocalDateTimeToUnix(value: string) {
  if (!value) {
    return null;
  }
  const timestampMs = Date.parse(value);
  if (Number.isNaN(timestampMs)) {
    return null;
  }
  return Math.floor(timestampMs / 1000);
}

function formatNumber(value: number | null) {
  if (value === null) {
    return "--";
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatScore(value: number) {
  return value.toFixed(3);
}

function formatMetric(value: number | null, unit: string) {
  const formatted = formatNumber(value);
  if (formatted === "--") {
    return formatted;
  }
  return `${formatted} ${unit}`;
}

function formatServing(entry: FoodEntry) {
  if (entry.servingSize) {
    return entry.servingSize;
  }
  if (entry.servingQuantity !== null) {
    return `${formatNumber(entry.servingQuantity)} ${entry.basis}`.trim();
  }
  return entry.basis || "--";
}

function truncateText(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 3)}...`;
}

function formatDuration(startTimestamp: number, endTimestamp: number) {
  const durationSeconds = Math.max(0, Math.round(endTimestamp - startTimestamp));
  if (durationSeconds < 60) {
    return `${durationSeconds}s`;
  }
  const minutes = Math.floor(durationSeconds / 60);
  const seconds = durationSeconds % 60;
  if (seconds === 0) {
    return `${minutes}m`;
  }
  return `${minutes}m ${seconds}s`;
}

function sectionHint(section: Section) {
  switch (section) {
    case "Dashboard":
      return "Recent activity";
    case "Memory":
      return "Search + clips";
    case "Faces":
      return "Identity review";
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
