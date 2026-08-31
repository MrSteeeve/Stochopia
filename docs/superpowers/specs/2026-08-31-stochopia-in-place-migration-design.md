# Stochopia In-Place Migration Design

**Date:** 2026-08-31  
**Status:** Approved
**Target repository:** `MrSteeeve/Stochopia`  
**Target local directory:** `<workspace-root>/Stochopia`

## Objective

Replace the project's legacy identity with Stochopia in the current repository, preserve the useful implementation and its Git history, and establish a clean `stochopia.*.v1` protocol line for the coming redesign.

Stochopia is positioned today as **A Stateful Environment for Training Trading-Desk Agents** and may later grow into **A World Model for Financial Decision Agents**. The durable brand line is:

> A constructed world for learning to act under uncertainty.

## Chosen Migration Strategy

Perform an in-place, intentionally breaking migration on `main`. Do not create a second repository and do not rewrite Git history. Rename the GitHub repository only after the local migration passes validation.

This strategy preserves authorship and development history while making the checked-out source tree, public interfaces, documentation, and current GitHub identity consistently Stochopia. Historical commits will continue to contain the former name; that is an explicit boundary, not a migration failure.

## Required Renames

The migration will apply consistently across source code, tests, schemas, scenarios, configuration, CI, documentation, and selected untracked work:

- Legacy Python package to `stochopia`.
- Legacy Python distribution to `stochopia-bench`.
- Legacy CLI to `stochopia-benchmark`.
- Project-facing classes and symbols containing the legacy brand to corresponding `Stochopia` names.
- Legacy CSI scenario directory to `scenarios/stochopia_csi`.
- Protocol, schema, manifest, seed, hash, and deterministic-sampling namespaces to a new `stochopia.*.v1` line.
- Environment variables and configuration keys containing the former project name to `STOCHOPIA_*` or `stochopia_*`, preserving their existing semantics.
- Repository title, examples, commands, paths, badges, comments, and current documentation to Stochopia terminology.
- File and directory names containing the former project name to Stochopia equivalents.

The selected untracked market-generator implementation and test are part of the migration and will be included in the resulting commit. Current untracked research notes under `docs/research/` will be preserved and reviewed individually. A reviewed research decision may be committed; full private conversation transcripts remain local-only.

## Compatibility Boundary

This is a new protocol identity, not a backward-compatible package alias:

- No compatibility shim for the legacy package import or executable will be retained.
- No old protocol namespace will be emitted by the new code.
- Existing legacy trajectories, manifests, recorded hashes, and deterministic seeds are not promised to replay under Stochopia.
- Stochopia fixtures and golden values affected by namespace-derived randomness will be regenerated and reviewed as `stochopia.*.v1` artifacts.
- No bulk replacement may alter unrelated third-party source material; replacements are scoped to the project's versioned working tree and selected source files.

## Preservation and Exclusions

The migration must preserve all pre-existing user work and must not silently discard or overwrite dirty-tree files.

The following are excluded from commits and repository-wide replacement:

- `blobs/`, including third-party PDFs, until provenance and redistribution rights are reviewed separately.
- `.env` values and all secrets. Only code/documentation references to variable names may be renamed.
- `.venv/`, caches, `.DS_Store`, build outputs, and existing distribution artifacts; generated artifacts will be rebuilt rather than renamed in place.
- `.git/` objects, refs, and commit history.
- Internal `refs/codex/turn-diffs/*`, which must never be mirror-pushed.

The repository remains private during this operation. Deleting or archiving another repository is outside this migration because the existing GitHub repository itself will be renamed.

## Execution Sequence

1. Record the dirty-tree inventory and current branch/remote state.
2. Rename package and scenario directories, including selected untracked source files.
3. Update project symbols, imports, CLI, packaging, schemas, protocol identifiers, configuration, tests, CI, documentation, and filenames.
4. Regenerate lock/build metadata only where the repository's normal tooling requires it.
5. Scan the versioned working tree and selected source files for residual case-insensitive occurrences of the former name.
6. Run targeted packaging, import, CLI, and protocol tests, then the full test suite.
7. Review the complete diff for accidental content changes, secret inclusion, third-party data, and unintended deletions.
8. Commit the migration on `main`, then rename the currently configured private GitHub repository to `MrSteeeve/Stochopia`.
9. Set `origin` to the canonical new URL, push only `main` through the normal branch push path, and verify remote visibility, default branch, and remote commit identity.
10. Rename the local directory from `eqd_simulation` to `Stochopia` only after remote verification, then rebuild the local virtual environment because its scripts and editable-package metadata contain absolute paths. The Codex task may need to be reopened at the new path.

If any test or remote operation fails, stop at the last recoverable state, keep all local changes, and report the exact failure rather than deleting, resetting, or forcing history.

## Acceptance Criteria

The migration is complete only when all of the following hold:

- The active source tree, tests, current docs, filenames, and selected untracked work contain no case-insensitive occurrence of the former project name, excluding `.git/`, `blobs/`, ignored environment/build/cache files, and historical Git objects.
- `import stochopia` succeeds from the built and editable package contexts used by the repository.
- `stochopia-benchmark --help` succeeds from a clean wheel smoke environment.
- Targeted tests and `.venv/bin/python -m pytest` pass.
- Packaging metadata exposes `stochopia-bench`, package `stochopia`, and CLI `stochopia-benchmark` only.
- Protocol/schema tests assert the new `stochopia.*.v1` identifiers and updated deterministic fixtures.
- The final diff contains no secrets, `blobs/`, caches, or unintended generated files.
- GitHub reports the private repository as `MrSteeeve/Stochopia`, `origin` uses its canonical URL, and `main` contains the migration commits.
- The local directory is `<workspace-root>/Stochopia`, its virtual environment has been rebuilt and smoke-tested there, or the only remaining action is reopening the Codex task after that final filesystem rename.
