# Automated Phishing Detection for Frontier AI Inference

## Overview

This repository contains the reproducible pipeline and supporting
documentation for a D.Eng. praxis on low-latency phishing detection for
frontier AI inference endpoints. The work emphasizes dataset provenance,
auditable feature extraction, baseline model evaluation, and deployment
latency validation.

## Quick Start

1. Install dependencies: `pip install -r code/requirements.txt`
2. Preprocess data: `python code/data_preprocessing.py --config data/configs/week1_snapshot_v2.yml`
3. Train baselines: `python code/train_baselines.py --config data/configs/week1_snapshot_v2.yml`
4. Evaluate metrics: `python code/evaluate_baselines.py --config data/configs/week1_snapshot_v2.yml`
5. Run ablations: `python code/run_feature_ablations.py --config data/configs/week1_snapshot_v2.yml`
6. Burst replay: `python code/burst_test.py --mode asgi --concurrency 64 --total 341 --test-file data/processed/v2025.12.10/test_data.csv --output data/processed/v2025.12.10/eval/burst_latency.json`

## Outputs

- Versioned artifacts: `data/processed/<snapshot>/` (dataset stats, metrics, eval summaries)
- Thesis-ready tables/figures: `reports/`
- Label governance logs: `data/labels/`

## Data Availability

Raw and processed datasets are not committed to the public repository.
Use `data/README.md` plus the placeholder notes in `data/raw/README.md`
and `data/processed/README.md` to populate sources and regenerate
artifacts.

## Manuscript

The final manuscript PDF/DOCX is distributed via the GitHub Releases
page for this repository.


## Citation
If you use this repository, please cite it using `CITATION.cff` (GitHub will surface this automatically on the repo landing page).

## License
This project is released under the MIT License (see `LICENSE`).
