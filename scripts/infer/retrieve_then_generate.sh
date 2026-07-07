#!/bin/bash
TEXT=${1:-"a dog running"}
IMAGE=${2:-""}
python scripts/infer/retrieval_demo.py --query "$TEXT" --top_k 3
python scripts/infer/generation_demo.py --text "$TEXT" --image "$IMAGE"
echo "Retrieve-then-Generate pipeline complete"
