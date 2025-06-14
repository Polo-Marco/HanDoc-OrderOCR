#!/bin/bash
set -e

mkdir -p models/det_model
mkdir -p models/rec_model
mkdir -p models/order_model

# Replace FILE_ID with the real file IDs from step 3

echo "Downloading detection model..."
gdown --id 15Dwnpnus_ferLV8H2ZKgKBWI07-uh2y4 -O models/det_model/inference.pdparams

echo "Downloading recognition model..."
gdown --id 1hVgSom4zAdYNLO0CTJKKkSjJ3PZ1kc3n -O models/rec_model/inference.pdparams

echo "Downloading reading order model..."
gdown --id 185NiiuUxQN8uUrZPJSUakFg7D9jc1qPd -O models/order_model/inference.pth

echo "All models downloaded!"

