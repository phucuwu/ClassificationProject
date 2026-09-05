# Task Checklist: Taste Profile Data Pipeline Integrity

## Phase 1: Contract and Dataset Database

- [ ] Task 1: Define label provenance, training eligibility, and temporal-effectiveness terminology in project documentation.
- [ ] Task 2: Add a tested idempotent Dataset database migration, legacy-row classification, and database constraints.
- [ ] Task 3: Restrict the Feature matrix and outlier analysis to reviewed human-confirmed Samples.
- [ ] **Checkpoint:** Verify provenance migration and training eligibility with focused database/API tests and a copy of the runtime Dataset database.

## Phase 2: Recording, Review, and Capture Integrity

- [ ] Task 4: Enforce label provenance and Sample validation at the Backend server recording and review boundaries.
- [ ] Task 5: Replace near-duplicate label consolidation with exact image-hash deduplication and non-destructive similarity warnings.
- [ ] Task 6: Remove desktop screen capture and make Userscript active-Artwork capture fail closed.
- [ ] **Checkpoint:** Verify Manual, Supervised, Full auto, review, and capture-failure flows end to end.

## Phase 3: Evaluation and Full Auto Warning

- [ ] Task 7: Replace the random holdout with a contiguous temporal holdout and remove in-sample effectiveness fallback.
- [ ] Task 8: Make temporal holdout metrics and the recall-first effectiveness warning authoritative in the model artifact and API.
- [ ] Task 9: Show provenance and model-trust state in the Developer dashboard; require acknowledgement before warning-state Full auto activation.
- [ ] **Checkpoint:** Verify temporal partitioning, metric labels, stale-model status, and Full auto acknowledgement in tests and browser behavior.

## Phase 4: Current Data Recovery

- [ ] Task 10: Back up, migrate, retrain, audit high-leverage errors, and document the current Dataset database results.
- [ ] **Complete:** Run `python -m pytest`, verify the migrated Dataset database, and confirm documentation matches runtime behavior.
