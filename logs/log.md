[2026-05-15T00:00:00Z] [implementer] feat(registry): plugin registry with TDD — 6/6 tests pass, mypy clean, SHA 0ddf85d
[2026-05-15T00:00:00Z] [implementer] task 4 complete: pydantic config schema implemented and tested (11 tests, mypy clean)
[2026-05-15T00:00:00Z] [implementer] fix(config): replaced multi-import block with single TrainConfig import; added test_all_public_submodels_are_importable; added min_length=1 to DataSplit.annotations/images; reformatted schema.py | commit 48f1bd6
[2026-05-15] [implementer] Task 14 complete — CLI Typer skeleton implemented and committed (b71c4f7)
[2026-05-15] [planner] scaffolding plan written and committed

[2026-05-15] [implementer] scaffolding complete; exit criteria pass
[2026-05-15] [implementer] post-review cleanup pass complete — 7 reviewer items fixed across 4 commits (13fe3f7)
[2026-05-16] [implementer] task 1 — added albumentations/opencv-headless deps, promoted pillow to core
[2026-05-16] [implementer] task 2 — appended deferred iscrowd + transform-suite TODO entries
[2026-05-16] [implementer] task 3 — added TextPromptMode + TextPromptConfig schema
[2026-05-16] [implementer] task 4 — added NormalizeConfig with range validator
[2026-05-16] [implementer] task 5 — added HFFieldMap schema with conventional defaults
[2026-05-16] [implementer] task 6 — added HFDatasetConfig schema
[2026-05-16] [implementer] task 7 — extended DataConfig with hf/text_prompt/normalize + format validator
[2026-05-16] [implementer] task 8 — implemented resolve_normalization
[2026-05-16] [implementer] task 9 — implemented build_eval_transforms
[2026-05-16] [implementer] task 10 — implemented build_train_transforms (det test uses compose.set_random_seed for alb 2.x)
[2026-05-16] [implementer] task 11 — implemented collate_batch
[2026-05-16] [implementer] task 12 — added COCO module-private helpers (index, remap, decode, prompts)
[2026-05-16] [implementer] task 13 — COCODataset full impl with prompt modes, iscrowd filter, multiplex cap (TDD, 17 new tests, 26 coco tests pass; only test_data_stubs fails as expected)
[2026-05-16] [implementer] task 14 — build_coco builder with pipeline + model_name kwargs (2 new tests; 28 coco tests pass)
[2026-05-16] [implementer] task 15 — hf module-private helpers (resolve_field, normalize_bbox, validate, class_names; 8 tests pass)
[2026-05-16] [implementer] task 16 — HFDataset + build_hf builder with field-map, masks-from-boxes fallback (8 new tests; 16 hf tests pass)
[2026-05-16] [implementer] task 17 — augment example YAMLs with text_prompt + normalize; ARCHITECTURE.md data line
[2026-05-16] [implementer] task 18 — boundary test guarding TrainConfig import in data layer
[2026-05-16] [implementer] task 19 — drop test_data_stubs and stale data imports
[2026-05-16] [implementer] task 20 — ruff/format clean sweep: per-file E402 for test_data_*, removed unused std vars, shortened docstring (135/135 pass, mypy clean)
[2026-05-16] [implementer] task 21 — coverage verification: 138 tests pass, total 91.36%, hf.py 81% after 3 added tests
