#!/usr/bin/env bash
# Runs the full pipeline. Use on a laptop or PC with an NVIDIA GPU (needed for the transformer).
set -e
python src/prepare_data.py
python src/baseline.py
python src/transformer_model.py
python src/evaluate.py
echo "Done. Send the 'results' folder (zip it) for the report."
