# Test Support Helpers

`tests/support/` contains shared scaffolding for test setup that would otherwise be copied across suites. Keep helpers small, explicit, and tied to stable test boundaries.

## Stable Helpers

Stable helper modules define `__all__`; adding to that list is a public test-support API change.

- `results.py` builds common `SimResult` shapes such as `sim_result()`, `failed_result()`, and `corner_results()`.
- `executors.py` provides synchronous fake executors for session, corner, and workspace tests.
- `backends.py` provides registry test backends and cleanup helpers.
- `ngspice.py` provides executable discovery and skip helpers. Use `ngspice_on_path` and `require_ngspice` from `conftest.py` in tests.
- `workspaces.py` provides small project and experiment builders.
- `assertions.py` holds public-contract assertions when the same check appears in more than one suite.

## Case Modules

Files ending in `_cases.py` are split-suite support for a specific topic. They may hold fixtures, sample circuits, local helper classes, or constants that came from a large test file split. Keep them owned by that topic instead of promoting them to stable helpers too early.

## When To Add A Helper

Add a helper when setup or assertions are repeated in multiple files, when the helper clarifies a public contract boundary, or when it keeps split test files focused on behavior rather than scaffolding.

## When Not To Add A Helper

Do not add a helper for one-off setup, hide important assertions behind a broad abstraction, or create generic builders with many optional parameters. Prefer direct test code until duplication is real.
