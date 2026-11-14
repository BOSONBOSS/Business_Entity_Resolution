# Data Directory

This folder manages dataset inputs and outputs for the Business Entity Resolution project.

## Directory Structure

- `raw/`: Symlink pointing to `student_resource/dataset/train/` (contains raw `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv`).
- `processed/`: Stores generated artifacts:
  - `benchmark_pairs.csv`: Curated 50,000 balanced pairs (25,000 true matches + 25,000 hard negatives).
  - `feature_matrix.csv`: 15 engineered features extracted per pair.
- `archives/`: Backup location for raw `.zip` bundles (gitignored due to GitHub 100MB file size limit).

## Data Access & Setup

The raw dataset contains 24.2 million records (~2.4 GB) and cannot be tracked directly in GitHub due to GitHub's 100 MB per-file upload limit. To set up the raw files locally, download the dataset bundle and place the TSV files into `student_resource/dataset/train/` and `student_resource/dataset/test/`.
