# See full spec at the parent repository

This file is a placeholder. The authoritative spec lives at:

`../../docs/superpowers/specs/2026-05-26-llm-feature-test-matrix-design.md`
(in the local-only parent workspace).

A public copy will be committed here once the local parent workspace has been
reviewed and we're ready to share the design openly. Until then, the README
in this repo summarises the intent.

## Why two copies?

The parent `llm-model-tests` workspace is local; this submodule is the public
research repo. The spec is currently mid-revision; copying it now would force
sync overhead. Once stable we'll vendor it into `docs/spec.md` for offline
readers.
