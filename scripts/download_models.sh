#!/usr/bin/env bash
# Fetch the weights maya-rs runs: maya-research/maya2-native (Apache-2.0) and kyutai/mimi (CC-BY-4.0).
# Needs the Hugging Face CLI (`pip install -U huggingface_hub`); about 5 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download maya-research/maya2-native --local-dir models/maya2-native \
  --include 'config.json' 'maya3_config.json' 'README.md' 'model*.safetensors*' 'voices/voices.json' 'backbone/*.json'
hf download kyutai/mimi model.safetensors --local-dir models/mimi
