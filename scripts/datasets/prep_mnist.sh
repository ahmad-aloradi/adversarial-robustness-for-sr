set -e
PYTHONPATH=$(pwd) PROJECT_ROOT=$PYTHONPATH python src/datamodules/components/vision/vision_prep.py --dataset mnist "$@"
