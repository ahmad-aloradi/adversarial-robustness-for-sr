set -e
PYTHONPATH=$(pwd) PROJECT_ROOT=$PYTHONPATH python src/datamodules/components/librispeech/librispeech_prep.py
