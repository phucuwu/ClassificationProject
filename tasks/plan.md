# Implementation Plan: Taste Profile Data Pipeline Integrity

## Objective

Make the Sample collection, review, training, evaluation, and automation pipeline trustworthy enough to assess and improve a Taste profile. The existing manually collected Samples remain intact. Only human-confirmed labels may train a model. Full auto mode remains available by user decision, but the Developer dashboard and Userscript must clearly warn when its latest temporal-holdout evaluation does not meet the agreed recall-first target.

## Confirmed Decisions

- Full auto mode remains available with an explicit warning, not a hard eligibility block.
- Remove the desktop screen-capture fallback and `/api/capture` endpoint. If the Userscript cannot identify the active Artwork Primary image, it must not record or rate that Artwork.
- A model meets the initial recall-first effectiveness target when its temporal holdout contains at least 30 Positive-class Samples, recall is at least 0.80, and precision is at least 0.60.
- The target is an operational warning threshold, not proof that a Taste profile is universally effective.

## Architecture Decisions

- Add explicit **label provenance** to every Sample. Valid values are `manual_rating`, `supervised_confirmation`, `review_confirmation`, and `auto_decision`. The glossary will define this term and its values before implementation.
- A **training-eligible Sample** is a reviewed Sample whose label provenance is `manual_rating`, `supervised_confirmation`, or `review_confirmation`. A Sample with `auto_decision` provenance never enters the Feature matrix until a human review changes its provenance to `review_confirmation`.
- Preserve old Sample rows during migration. Existing Manual-mode reviewed Samples become `manual_rating`; existing Supervised-mode reviewed Samples become `supervised_confirmation`; existing Auto-mode Samples are `auto_decision` and must be unreviewed, because historical human confirmation cannot be inferred safely.
- Keep one `samples` table. SQLite migrations must rebuild the table transactionally when needed so new `CHECK` constraints apply to both fresh and existing Dataset databases.
- Validate API inputs at the Backend server boundary and enforce the same invariants in the Dataset database: binary or null label, finite Prediction score in `[0, 1]`, valid operating mode, binary review status, valid label provenance, and a 3,072-byte Vision embedding.
- Replace visual-similarity consolidation with exact Primary-image hash deduplication. Vision-embedding similarity is advisory data for the Developer dashboard and never changes a human label.
- Use Sample creation order for a contiguous temporal holdout. The newest suffix is the holdout; the earlier prefix is the development partition. Do not shuffle either partition across time. If the chosen holdout cannot retain both classes and the configured minimum Positive-class count, report that temporal evaluation is unavailable rather than silently falling back to a random split.
- Use development-partition cross-validation only to select Logistic Regression parameters and calibrate the Decision threshold. The temporal holdout is the only model-effectiveness report and the source of Full auto warning state.
- The final classifier may fit all training-eligible Samples after temporal evaluation, but its model artifact must retain the temporal holdout boundary, metrics, effectiveness state, threshold provenance, and training-eligibility counts.

## Data Migration and Recovery

1. Back up `data/dataset.db` and `data/model.json` before running the migration.
2. Add provenance and constraints through a tested, idempotent Dataset database migration.
3. Preserve the 3,305 existing Manual-mode reviewed Samples as training-eligible `manual_rating` Samples.
4. Mark every historical Auto-mode Sample as unreviewed `auto_decision`, even if its legacy `reviewed` value was `1`; it cannot be trusted as human-confirmed from the existing schema.
5. Invalidate the saved model artifact after schema migration because it predates the complete current Dataset database and lacks provenance-aware temporal evaluation metadata.
6. Retrain from the migrated Manual-mode Samples, review outlier candidates, then perform a new temporal evaluation.

## Dependency Graph

```text
Glossary and data-pipeline contract
    |
Dataset database migration and invariants
    |
    +-- API request validation and provenance-aware review/recording
    |       |
    |       +-- Userscript collection identity and Full auto warning
    |       +-- Developer dashboard provenance and effectiveness display
    |
    +-- Training-eligible Feature matrix
            |
            +-- Temporal evaluation and model artifact
                    |
                    +-- Documentation and current-data retraining run
```

## Task List

### Phase 1: Contract and Dataset Database

#### Task 1: Define provenance and effectiveness terms

**Description:** Extend the canonical glossary and architecture documentation with label provenance, training eligibility, temporal holdout, and effectiveness-warning definitions. Correct the existing Full auto review description to match the required behavior.

**Acceptance criteria:**
- [ ] Every new data-pipeline term is defined in `agent-docs/glossary.md`.
- [ ] Architecture documentation states that Full auto Samples are unreviewed until a human review.
- [ ] Documentation states that the temporal holdout, not selected cross-validation metrics, is the effectiveness report.

**Verification:**
- [ ] Review all new terms against `AGENTS.md` terminology rules.
- [ ] Confirm README and architecture descriptions do not claim random splitting is chronological.

**Dependencies:** None.

**Files likely touched:**
- `agent-docs/glossary.md`
- `agent-docs/system-architecture.md`
- `README.md`

**Estimated scope:** Small.

#### Task 2: Add an idempotent Dataset database migration and invariants

**Description:** Create the schema versioning/migration path that introduces `label_provenance`, enforces Sample invariants, and safely classifies legacy rows without losing Primary images, Vision embeddings, labels, or creation times.

**Acceptance criteria:**
- [ ] Fresh Dataset databases enforce valid label, Prediction score, operating mode, review status, label provenance, and Vision-embedding byte length.
- [ ] Migrating a legacy Dataset database preserves row IDs, image hashes, paths, labels, embeddings, and timestamps.
- [ ] Existing reviewed Manual-mode and Supervised-mode Samples receive human-confirmed provenance.
- [ ] Existing Auto-mode Samples become `auto_decision` and `reviewed = 0`.
- [ ] The migration is repeatable without changing already migrated rows.

**Verification:**
- [ ] `python -m pytest tests/test_database.py`
- [ ] New migration tests exercise fresh schema, legacy migration, invalid writes, and repeat migration.
- [ ] Run SQLite `PRAGMA integrity_check` against a copy of `data/dataset.db`.

**Dependencies:** Task 1.

**Files likely touched:**
- `backend/database.py`
- `tests/test_database.py`

**Estimated scope:** Medium.

#### Task 3: Restrict the Feature matrix to training-eligible Samples

**Description:** Change training data loading and statistics so they distinguish all Samples from human-confirmed, training-eligible Samples. Ensure label provenance is returned where the Review queue and Dataset inspector need it.

**Acceptance criteria:**
- [ ] `load_training_matrix()` returns only training-eligible Samples by default.
- [ ] A reviewed auto-decision is excluded until a review explicitly changes provenance to `review_confirmation`.
- [ ] Dataset statistics show total Samples, training-eligible Samples, and counts by provenance without changing existing aggregate semantics unexpectedly.
- [ ] Outlier analysis uses the same training-eligible matrix as model training.

**Verification:**
- [ ] `python -m pytest tests/test_database.py tests/test_phase8.py`
- [ ] Tests prove auto decisions cannot enter the Feature matrix before review and can enter after review confirmation.

**Dependencies:** Task 2.

**Files likely touched:**
- `backend/database.py`
- `backend/app.py`
- `tests/test_database.py`
- `tests/test_phase8.py`

**Estimated scope:** Medium.

### Checkpoint: Trusted Training Inputs

- [ ] Run `python -m pytest tests/test_database.py tests/test_api.py tests/test_phase8.py`.
- [ ] Inspect a copy of the current Dataset database: all current Manual-mode Samples are training-eligible; no Auto-mode Sample is training-eligible without review provenance.
- [ ] Review the migration backup and rollback procedure before applying it to runtime data.

### Phase 2: Recording, Review, and Capture Integrity

#### Task 4: Enforce provenance at API boundaries and review transitions

**Description:** Make the Backend server, not the caller, assign label provenance and review state. Validate all Sample-related request fields with constrained Pydantic types. A review action must explicitly convert an auto decision to `review_confirmation`.

**Acceptance criteria:**
- [ ] `/api/record` derives provenance from the operating mode and rejects invalid labels, modes, scores, and review states.
- [ ] Every Full auto Sample persists as `auto_decision`, `reviewed = 0`.
- [ ] Manual and confirmed Supervised-mode Samples persist as reviewed human-confirmed Samples.
- [ ] `/api/review` atomically writes the selected label, `reviewed = 1`, and `review_confirmation` provenance.
- [ ] API responses expose provenance where needed without breaking existing Userscript and dashboard callers.

**Verification:**
- [ ] `python -m pytest tests/test_api.py`
- [ ] Add API tests for invalid payloads, auto-label exclusion, and review-to-training eligibility transition.

**Dependencies:** Tasks 2-3.

**Files likely touched:**
- `backend/app.py`
- `backend/database.py`
- `tests/test_api.py`

**Estimated scope:** Medium.

#### Task 5: Remove destructive near-duplicate consolidation

**Description:** Limit deduplication to an identical image hash. Preserve visually similar Samples and expose similarity only as a non-destructive review signal.

**Acceptance criteria:**
- [ ] Re-recording the exact Primary image does not create a second Sample or overwrite a confirmed label.
- [ ] A non-identical Primary image with similarity at or above 0.98 creates a separate Sample and retains both labels.
- [ ] Similarity warnings do not alter label, mode, review state, or provenance.

**Verification:**
- [ ] `python -m pytest tests/test_phase8.py tests/test_api.py`
- [ ] Add regression tests for same hash, near visual match, and conflicting human ratings.

**Dependencies:** Task 4.

**Files likely touched:**
- `backend/app.py`
- `backend/database.py`
- `tests/test_phase8.py`
- `tests/test_api.py`

**Estimated scope:** Small.

#### Task 6: Make Userscript capture fail closed and remove screen capture

**Description:** Delete the desktop screen-capture endpoint and fallback. Narrow active Artwork extraction to an explicit active-card strategy and require the captured Primary-image element to remain connected, visible, and associated with the active card immediately before recording and dispatching a rating action. Remove broad cross-origin permissions that existed only to support unrestricted image and desktop capture.

**Acceptance criteria:**
- [ ] `/api/capture`, `mss` usage, and Userscript desktop-capture calls are removed.
- [ ] The Userscript no longer declares `@connect *`, and Backend-server browser CORS allows only the local Developer dashboard origin(s) required by the application.
- [ ] If the active card or Primary image cannot be validated, the Userscript shows/logs a failure and sends neither `/api/record` nor a rating key signal.
- [ ] Manual, Supervised, and Full auto flows revalidate the captured Artwork before recording and action dispatch.
- [ ] The extraction scope never falls back to global `body` or `main`.

**Verification:**
- [ ] `python -m pytest tests/test_api.py`
- [ ] Add focused static Userscript checks or a browser test harness for no global fallback and no capture endpoint use.
- [ ] Manually test each mode against the Library and record known DOM selector assumptions.

**Dependencies:** Task 4.

**Files likely touched:**
- `userscript/taste_collector.user.js`
- `backend/app.py`
- `requirements.txt`
- `tests/test_api.py`
- `README.md`

**Estimated scope:** Medium.

### Checkpoint: Correct Collection Lifecycle

- [ ] Run `python -m pytest tests/test_database.py tests/test_api.py tests/test_phase8.py`.
- [ ] Verify one Manual-mode and one Supervised-mode Sample become training-eligible.
- [ ] Verify one Full-auto Sample appears in the Review queue, is excluded from training, then enters training only after an explicit review.
- [ ] Verify an unidentifiable active Artwork causes no rate action and no Sample record.

### Phase 3: Evaluation and Automation Warning

#### Task 7: Replace random holdout with a contiguous temporal holdout

**Description:** Partition training-eligible Samples by creation order. Reserve the newest valid suffix for holdout and use the earlier prefix for development. Remove all in-sample evaluation fallback reporting as a model-effectiveness signal.

**Acceptance criteria:**
- [ ] Temporal holdout Samples are newer than every development Sample.
- [ ] The model returns `temporal_evaluation_unavailable` when a valid holdout cannot meet both-class and minimum-positive requirements; it never substitutes a random split.
- [ ] Training rejects insufficient training data for model use instead of producing in-sample effectiveness metrics.
- [ ] Tests cover ordered Samples, insufficient recent Likes, boundary selection, and no leakage between partitions.

**Verification:**
- [ ] `python -m pytest tests/test_model.py`
- [ ] New tests assert exact development/holdout IDs and timestamps for deterministic synthetic data.

**Dependencies:** Task 3.

**Files likely touched:**
- `backend/model.py`
- `tests/test_model.py`

**Estimated scope:** Medium.

#### Task 8: Make temporal metrics and effectiveness warning authoritative

**Description:** Separate tuning metrics from effectiveness metrics in the model artifact and API. Apply the agreed temporal-holdout warning target: at least 30 holdout Likes, recall >= 0.80, precision >= 0.60.

**Acceptance criteria:**
- [ ] Cross-validation metrics are labelled as development/tuning metrics.
- [ ] Temporal-holdout metrics are labelled as model effectiveness metrics.
- [ ] The artifact stores evaluation boundary, holdout counts, warning reasons, threshold source, and training-eligible counts.
- [ ] The warning is active when temporal evaluation is unavailable or any agreed target is unmet.
- [ ] The final classifier may still fit all eligible data only after temporal evaluation is recorded.

**Verification:**
- [ ] `python -m pytest tests/test_model.py tests/test_phase10.py`
- [ ] Add tests for every warning reason and for a model that meets all targets.

**Dependencies:** Task 7.

**Files likely touched:**
- `backend/model.py`
- `backend/app.py`
- `tests/test_model.py`
- `tests/test_phase10.py`

**Estimated scope:** Medium.

#### Task 9: Surface data trust and Full auto warnings in the Developer dashboard and Userscript

**Description:** Show training-eligible counts, label provenance, temporal evaluation status, warning reasons, and the distinction between tuning and effectiveness metrics. Require explicit acknowledgement of the warning each time Full auto mode is enabled.

**Acceptance criteria:**
- [ ] The Developer dashboard identifies whether the saved model is stale relative to the training-eligible Dataset database.
- [ ] The Developer dashboard presents temporal effectiveness metrics separately from cross-validation tuning metrics.
- [ ] The Review queue and Dataset inspector display label provenance.
- [ ] The Userscript warns and requires explicit acknowledgement before enabling Full auto when effectiveness is unavailable or below target.
- [ ] Full auto remains usable after acknowledgement, as selected by the user.

**Verification:**
- [ ] `python -m pytest tests/test_dashboard.py tests/test_api.py`
- [ ] Browser-test dashboard states and Full-auto warning/acknowledgement behavior against stubbed API responses.

**Dependencies:** Tasks 4 and 8.

**Files likely touched:**
- `backend/static/index.html`
- `backend/static/app.js`
- `backend/static/style.css`
- `userscript/taste_collector.user.js`
- `backend/app.py`
- `tests/test_dashboard.py`
- `tests/test_api.py`

**Estimated scope:** Medium.

### Checkpoint: Honest Model Reporting

- [ ] Run `python -m pytest tests/test_model.py tests/test_phase10.py tests/test_api.py tests/test_dashboard.py`.
- [ ] Verify temporal evaluation uses only later Samples and no random splitter remains in the training path.
- [ ] Verify Full auto requires warning acknowledgement but remains available.

### Phase 4: Data Recovery and Release Verification

#### Task 10: Migrate, retrain, and audit the current collection

**Description:** Apply the tested migration to the local runtime Dataset database, invalidate the old model artifact, retrain from all eligible Manual-mode Samples, and audit high-leverage model errors before using automation.

**Acceptance criteria:**
- [ ] A timestamped backup exists before migration.
- [ ] The migrated Dataset database passes integrity checks and preserves current Manual-mode Sample counts.
- [ ] A new artifact is trained from all current training-eligible Samples and records temporal effectiveness status.
- [ ] Review candidates include positive-class outliers and temporal-holdout false-positive Like decisions.
- [ ] The old stale artifact is not presented as current model status.
- [ ] Test dependencies are declared in a reproducible development dependency file or documented install command so a fresh environment can run the specified verification commands.

**Verification:**
- [ ] `python -m pytest`
- [ ] Run the migration and `PRAGMA integrity_check` against the local Dataset database.
- [ ] Record pre/post Sample, Like, Dislike, eligible, and provenance counts in a release note.
- [ ] Manually inspect the audit queue before enabling Full auto.

**Dependencies:** Tasks 1-9.

**Files likely touched:**
- `backend/database.py`
- `backend/model.py`
- `backend/app.py`
- `README.md`
- `agent-docs/system-architecture.md`

**Estimated scope:** Small code change plus operational runbook.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Existing Auto-mode history lacks reliable human-confirmation provenance | High | Exclude it from training and reset it to unreviewed auto-decision status during migration. |
| Temporal holdout has fewer than 30 Positive-class Samples | High | Show explicit effectiveness-unavailable warning; continue Manual and Supervised modes; collect later reviewed Samples. |
| Removing screen capture prevents collection on a Library DOM change | Medium | Fail closed, log the selector failure, and update the active-card extractor rather than collecting uncertain labels. |
| SQLite table rebuild migration risks runtime data | High | Back up first, test the migration against a copy, transact the rebuild, and verify with `PRAGMA integrity_check`. |
| Full auto remains user-permitted below target | High | Require acknowledgement on every activation; keep all auto decisions unreviewed and excluded from training until review. |
| Threshold choice may overfit development data | Medium | Label development metrics clearly and make temporal-holdout results authoritative. |

## Out of Scope

- Replacing CLIP or Logistic Regression with a different model family.
- Claiming that the current collection objectively measures a user's preference quality.
- Adding remote synchronization, authentication, or multi-user workflows.

## Completion Criteria

- The complete test suite passes: `python -m pytest`.
- The Dataset database preserves existing Manual-mode Samples and prevents unconfirmed automated labels from entering training.
- The model reports temporal, not random or in-sample, effectiveness metrics.
- The Developer dashboard and Userscript communicate model trust state before Full auto use.
- Documentation matches actual collection, review, training, and evaluation behavior.
