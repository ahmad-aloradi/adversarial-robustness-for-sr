set -e
PYTHONPATH=$(pwd) PROJECT_ROOT=$PYTHONPATH python src/datamodules/components/vision/tiny_imagenet_prep.py "$@"
