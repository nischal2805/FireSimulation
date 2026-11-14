## Task 12: P0-012 Documentation stubs

**Files:** `docs/ARCHITECTURE.md`, `docs/NUMERICAL_METHODS.md`, `docs/COORDINATE_CONVENTIONS.md`,
`docs/TEST_STRATEGY.md`, `docs/PERFORMANCE_BUDGET.md`, `README.md` (update).

- Each file points to the PRD section it implements (no duplicated PRD text) and records only what the code adds:
  ARCHITECTURE: module map (actual files), lifecycle, ownership (snapshot host read-only, device mirrors owned by
  backends), and the PRD 6.2 per-field table: units, memory layout (actual array shapes, MAC faces), owner,
  initialisation. NUMERICAL_METHODS: list ADR-0002/0003 and the controller rulings (edge-flux conduction, sampling
  offset, sub-stepping), parameter files. COORDINATE_CONVENTIONS: pointer to PRD 4 plus the pixel-centre convention
  and `look_at` construction. TEST_STRATEGY: test files mapped to PRD 10 items and markers. PERFORMANCE_BUDGET:
  pointer to PRD 5.4/11, current CPU baseline CSV path, VRAM table "pending GPU run".
- README: setup, commands, CPU-only note.

**Acceptance:** links resolve; no PRD paragraphs copied.

