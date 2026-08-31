# Stochopia In-Place Migration Implementation Plan

**Design:** `docs/superpowers/specs/2026-08-31-stochopia-in-place-migration-design.md`  
**Branch:** `main` (explicitly authorized by the user)  
**Delivery:** rename the existing private GitHub repository, push `main`, then rename the local directory

## Task 1: Freeze and Protect the Starting State

1. Record `git status`, `HEAD`, `origin/main`, remotes, tracked files, and untracked files.
2. Confirm `blobs/`, secrets, virtual environments, caches, and build outputs are excluded.
3. Confirm the selected untracked market-generator source, test, and research notes remain present.
4. Do not reset, clean, mirror-push, or rewrite history.

## Task 2: Rename the Python and Scenario Surfaces

1. Rename the legacy Python package directory to `stochopia/`.
2. Rename the legacy CSI scenario directory to `scenarios/stochopia_csi/`.
3. Rename any filenames containing the legacy brand.
4. Update imports, resource-package lookups, project-facing class names, CLI names, distribution metadata, and package declarations.
5. Update environment-variable names without reading, printing, or committing secret values.

## Task 3: Establish the Stochopia v1 Protocol Identity

1. Replace protocol, schema, manifest, trajectory, evaluation, and experiment identifiers with `stochopia.*.v1` identifiers.
2. Replace deterministic seed, cache, and sampling namespaces with `stochopia.seed.<purpose>.v1` or another explicit `stochopia.<kind>.v1` identifier.
3. Update scenario configuration, task-suite examples, and fixtures.
4. Regenerate affected deterministic expected values through the repository's tests rather than preserving legacy hashes by alias.

## Task 4: Migrate Tests, CI, and Documentation

1. Update all test imports, class references, fixture paths, schema assertions, and CLI assertions.
2. Update CI wheel smoke checks to the new distribution, package, and CLI.
3. Update both READMEs, protocol/data-audit/redesign documents, design records, current research notes, and repository tree examples.
4. Include the selected untracked market generator, its test, and the reviewed research decision; retain the full private conversation transcript locally only.
5. Keep `blobs/` and ignored local artifacts untracked.

## Task 5: Validate the Migration

1. Scan tracked and selected source paths for case-insensitive legacy-brand occurrences; require zero matches.
2. Run `git diff --check` and inspect rename/delete/add statistics.
3. Run focused import, CLI, schema/protocol, market-generator, and packaging tests.
4. Run `.venv/bin/python -m pytest`.
5. Build wheel/sdist and install the wheel into a clean temporary virtual environment.
6. Verify `import stochopia` and `stochopia-benchmark --help` from the clean environment.
7. Inspect the staged manifest for secrets, `blobs/`, caches, local settings, and unintended generated files.

## Task 6: Commit and Publish

1. Commit the source migration on `main` without amending or rewriting prior commits.
2. Rename the existing private GitHub repository to `MrSteeeve/Stochopia`.
3. Set `origin` to `https://github.com/MrSteeeve/Stochopia.git`.
4. Push only `main` through the normal branch push path.
5. Verify the remote repository is private, its default branch is `main`, and remote `main` equals local `HEAD`.

## Task 7: Rename the Local Directory

1. After remote verification, rename the local `eqd_simulation` directory to sibling directory `Stochopia`.
2. Recreate `.venv` at the new path and reinstall the locked project so scripts, editable-package metadata, and direct URLs no longer reference the old absolute path.
3. Verify tests or a focused post-move smoke check, repository status, remote, and `HEAD` from the new path.
4. Report that the current Codex task may need to be reopened against the renamed directory.

## Stop Conditions

- Stop without destructive cleanup if a required test repeatedly fails.
- Stop before the GitHub rename if the migration diff includes secrets, third-party blobs, or unexplained deletions.
- Stop before the local-directory rename unless the new GitHub name and pushed `main` are verified.
- Preserve all local changes if a remote operation fails; never force-push.
