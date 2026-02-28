# Website Frontend Spec

## Purpose

The `/website` app is the laptop-oriented companion UI for Cleo.

It is separate from `/frontend`, which is the low-latency display surface for the AR glasses. The website is the place where a user can:

- review data the system has collected
- browse notes and summaries
- inspect video and transcription history
- manage app integrations
- edit user preferences and accessibility settings

This interface should feel like an admin console plus personal memory dashboard, optimized for desktop and laptop screens first, with responsive support for tablets.

## Product Goals

- Give the user a clear, searchable view of what Cleo has captured and stored.
- Make settings and preferences easy to discover and update.
- Surface useful summaries instead of forcing the user to inspect raw logs.
- Provide enough structure to grow into a full control panel for the device and services.
- Present data in a polished, modern UI that feels intentionally designed rather than like a debugging tool.

## Non-Goals

- This website is not the in-glasses runtime UI.
- This website does not need to mirror the HUD interaction model.
- This website should not directly handle real-time sensor streaming in v1.
- This website does not need full device setup / onboarding in the first version unless it becomes necessary later.

## Primary Users

- The device owner reviewing their captured history and notes.
- A developer / power user configuring preferences, enabled apps, and system behavior.

## Core User Jobs

1. Change personal preferences such as accessibility and display-related settings.
2. Review recent transcriptions and search past conversations or observations.
3. Review note summaries generated from prior activity.
4. Review captured video clips associated with moments in time.
5. Inspect food macro entries captured by the system.
6. Enable, disable, and inspect registered apps/tools.

## Information Architecture

Recommended top-level navigation:

1. Dashboard
2. Memory
3. Notes
4. Nutrition
5. Apps
6. Preferences
7. System (optional in v1, if implementation is cheap)

## Screens

### 1. Dashboard

Purpose: give the user a quick view of recent activity and useful entry points.

Include:

- summary cards for recent transcriptions, notes, video clips, and enabled apps
- recent activity feed ordered by time
- quick actions: search memory, review notes, edit preferences
- system health summary (service reachable / unavailable) if available

This page should answer: "What has Cleo captured recently, and where should I go next?"

### 2. Memory

Purpose: browse and search stored history.

This section should combine:

- transcription history
- semantic search results
- video clip references tied to time ranges

Views:

- searchable transcription table/list
- global search box for text search
- result detail panel showing timestamp, confidence, related clips, and adjacent entries
- time-range filtering

Mapped backend data:

- `GetTranscriptionLog`
- `GetTranscriptionsInRange`
- `Search`
- `GetVideoClipsInRange`
- `GetVideoClip` (for clip playback/download)

### 3. Notes

Purpose: provide a clean reading experience for generated note summaries.

Include:

- paginated list of note summaries
- note detail view
- filtering by date / time range
- ability to jump from a note to related transcriptions or video clips in the same time window

Mapped backend data:

- `GetNoteSummaries`
- optionally `GetTranscriptionsInRange`
- optionally `GetVideoClipsInRange`

### 4. Nutrition

Purpose: display captured food macro information in an easy-to-scan format.

Initial requirements:

- table view of food macros
- columns for product name, brand, calories, protein, fat, carbs, serving size, recorded time
- sorting and filtering
- empty state when no data exists

Important note:

- The website now reads nutrition entries through `GET /api/food-macros`.
- That endpoint currently reads from the shared SQLite database through the website BFF.
- Longer term, this should move behind a dedicated `DataService` read/list RPC so the HTTP layer can proxy the gRPC contract instead of reading SQLite directly.

### 5. Apps

Purpose: manage registered apps and tool integrations.

Include:

- list of registered apps
- filters by app type / enabled state
- app detail drawer or page
- enable / disable toggles
- display of description, gRPC address, schema, and status

Mapped backend data:

- `ListApps`
- `SetAppEnabled`

Optional future enhancement:

- app registration UI if exposing `RegisterApp` through an admin flow makes sense

### 6. Preferences

Purpose: let the user manage persistent preferences.

Initial settings to support:

- color blindness type
- preferred contrast mode
- text size / UI density
- notification verbosity
- default time range for history views

Technical note:

- `DataService` currently stores preferences as key/value strings via `GetPreference` and `SetPreference`.
- The website should treat preferences as a structured client-side schema mapped onto those string keys.

Example preference keys:

- `accessibility.color_blindness_type`
- `accessibility.contrast_mode`
- `ui.text_scale`
- `ui.density`
- `notifications.verbosity`
- `memory.default_time_range`

### 7. System (Optional)

Purpose: lightweight operational visibility.

Include if low-cost:

- service connectivity indicators
- app/backend version display
- links to logs or diagnostics

This is useful for development and power users, but not required for the first usable release.

## UX Requirements

- Desktop-first layout with responsive behavior down to tablet width.
- Fast navigation between list and detail views.
- Strong search UX with obvious filters and clear empty states.
- Time-based data should be easy to scan and sort.
- Tables should support pagination, sorting, and row selection.
- Video clips should open in an inline modal or side panel instead of forcing page navigation.
- Accessibility settings should be reflected in the website itself where feasible.

## Visual Direction

The site should feel like a premium personal control center, not a developer dashboard template.

Design guidance:

- clean, high-contrast information layout
- strong typography and spacing
- card + panel composition for summary screens
- table-heavy views for logs, but balanced with visual hierarchy
- restrained but purposeful motion
- clear empty states and loading states

Avoid:

- overusing debug-style raw JSON views
- crowded enterprise admin visuals
- making the UI resemble the glasses HUD

## Local Run

Use Bazel to start the website, and use `pnpm` as the package manager:

```bash
bazel run //:website_run
```

This launcher starts the Vite dev server and the local website API used for `/api/*` requests.

To install website dependencies without starting the dev server:

```bash
bazel run //:website_install
```

## Data Integration Constraints

Important architecture constraint:

- Browsers should not talk directly to raw Python gRPC services unless we intentionally add `grpc-web` support plus a proxy layer.

Recommended approach:

- build the website as a web app that talks to a small HTTP/JSON backend-for-frontend (BFF)
- the BFF can live in the existing Python backend or as a dedicated service
- the BFF translates browser requests into calls to `DataService`

Why:

- simpler browser compatibility
- easier auth/session handling later
- easier data shaping for UI needs
- avoids forcing the frontend to speak native gRPC

## API Requirements For The Website

The website can be built now against a thin API layer that wraps the current gRPC methods.

Suggested initial HTTP endpoints:

- `GET /api/dashboard/summary`
- `GET /api/transcriptions`
- `GET /api/transcriptions/range`
- `GET /api/search`
- `GET /api/videos`
- `GET /api/videos/:id`
- `GET /api/notes`
- `GET /api/apps`
- `POST /api/apps/:name/enabled`
- `GET /api/preferences/:key`
- `POST /api/preferences/:key`

Backend gaps to add soon:

- corresponding `DataService` read RPC for food macros so the website BFF no longer needs direct SQLite reads

## Recommended Frontend Stack

Recommended framework: **Next.js (App Router) with TypeScript**

Why this is the best fit:

- excellent for building polished, production-quality web apps quickly
- supports server-side data fetching and route handlers out of the box
- makes it easy to add an internal BFF layer for the browser
- mature ecosystem for tables, forms, charts, auth, and design systems
- easy to build a desktop-first app that still degrades well on smaller screens

Recommended supporting libraries:

- `Tailwind CSS` for fast, consistent styling
- `shadcn/ui` for accessible, customizable primitives
- `TanStack Query` for client-side caching and async state
- `TanStack Table` for transcription, notes, apps, and nutrition tables
- `Zod` for API payload validation
- `React Hook Form` for preferences and settings forms

This stack is a strong fit because the website needs:

- data-heavy views
- settings forms
- clean component composition
- polished UI with minimal friction
- room to grow into auth, multi-page navigation, and richer admin workflows

## Why Not Reuse `/frontend`

The existing `/frontend` is for glasses presentation and likely has different constraints:

- lower-latency display assumptions
- different layout and interaction model
- different information density

The laptop website should be treated as a separate product surface with its own UX and routing, even if some shared components or API utilities are reused later.

## Suggested v1 Scope

Ship the first website with:

1. Dashboard
2. Memory (transcriptions + search + clips)
3. Notes
4. Apps
5. Preferences

Include Nutrition in v1. It already reads from the local database through the website API layer.

## Implementation Notes

- Keep `/website` as a standalone frontend app.
- Prefer a typed API client layer between React components and backend calls.
- Normalize timestamps at the API boundary and format them consistently in the UI.
- Design around loading, empty, and error states from day one.
- Use mock data first for layout if some backend endpoints are not ready.

## Backend Follow-Ups Needed

1. Add read/list support for food macros to `DataService`.
2. Expand the HTTP/JSON layer (or add a `grpc-web` gateway) for the rest of the browser-facing endpoints.
3. Optionally add combined dashboard summary endpoints so the UI does not need to fan out across many requests.

## Success Criteria

The website is successful when a user can:

- update preferences without touching code
- inspect recent and historical transcriptions easily
- read note summaries in a clean interface
- find related video clips for a time window
- manage enabled apps
- understand what data Cleo has collected without using developer tools
