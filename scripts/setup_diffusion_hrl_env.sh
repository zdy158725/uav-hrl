#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hhmarl_diffusion}"
conda create -y -n "${ENV_NAME}" --clone hhmarl
conda run -n "${ENV_NAME}" python -m pip install \
  diffusers==0.11.1 \
  einops==0.4.1 \
  huggingface-hub==0.13.4

echo "Environment ready: ${ENV_NAME}"
