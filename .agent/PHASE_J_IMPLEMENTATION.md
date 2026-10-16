# Phase J — Responsive Behaviour (Implemented)

- ≥1400px: three columns (292 / flexible / 336).
- 1100–1400px: tightened rails (250 / flexible / 290).
- <1100px: single column; the project sidebar and the activity rail become
  **overlay drawers** (fixed, dismissible, toggled from the header), and the chat
  remains the primary surface. A floating "💬 Agent" button keeps the agent
  reachable from every page on wide screens.
- ≤640px: condensed header, stacked form rows, narrower table padding, shorter
  feature-bar labels.
- All data tables (experiments, comparison, features, provenance, report) are
  horizontally scrollable instead of clipped.
- SVG charts scale via `viewBox`; metric bars degrade to plain numbers.
- `prefers-reduced-motion: reduce` disables transitions/animations.
- Sidebar, header and rail wrap instead of overflowing (the pre-existing
  header-overflow bug).
