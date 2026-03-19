#!/bin/bash

set -euo pipefail


PYTHONPATH=$(pwd) python src/datamodules/components/asvspoof5/asvspoof_prep.py

