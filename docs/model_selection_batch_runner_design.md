# Model Selection Batch Runner Design

The next training runner should execute explicit jobs from `configs/model_selection_batch_A.yaml` or `configs/model_selection_batch_B.yaml`.

Required behavior:

- Use stable `job_id` for resumability; generate a separate timestamped `run_id` for artifacts.
- Skip completed `job_id` entries by default.
- Support `--force-jobs JOB_ID[,JOB_ID...]` and `--force-all`.
- Refuse to start a new job if remaining walltime is less than `estimated_minutes_per_job + walltime_safety_margin_minutes`.
- Include `dataset_version`, source dataset hash, split policy, model config, seed, loss-weighting mode, and artifact policy in every manifest.
- Keep large files out of GitHub: no `.pt`, checkpoints, zips, large CSVs, predictions, or full result folders.
- Treat Batch B as disabled until Batch A and the targeted25k audit are reviewed.
