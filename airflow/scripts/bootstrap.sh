#!/bin/bash
set -e

echo "Installing Git and DVC..."
pip install "dvc[gcs]" git-remote-https
