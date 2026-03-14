#!/bin/bash
# CNCeleb dataset preparation script

set -euo pipefail

# Default ROOT_DIR
default_path="data/cnceleb"
if [ -d "data/datasets/cnceleb" ]; then
    default_path="data/datasets/cnceleb"
fi

stage=4
stop_stage=4

ROOT_DIR="${1:-$default_path}"

# set to true to download MUSAN and RIRs for data augmentation
include_augmented_data=false

# Concatenation parameters
TARGET_DURATION="${2:-5.0}"
MIN_THRESHOLD="${3:-2.0}"
OUTPUT_DIR="${4:-$ROOT_DIR/concatenated}"
MAPPING_FILE="${5:-$ROOT_DIR/metadata/concat_mapping.map}"

# Dataset subdirectory names
CNCELEB1_FLAC="CN-Celeb_flac"
CNCELEB2_FLAC="CN-Celeb2_flac"
CNCELEB1_WAV="CN-Celeb_wav"
CNCELEB2_WAV="CN-Celeb2_wav"

num_jobs=16

if [ $include_augmented_data = "true" ]; then
        downloadables_list="musan.tar.gz rirs_noises.zip cn-celeb_v2.tar.gz cn-celeb2_v2.tar.gz"
    else
        downloadables_list="cn-celeb_v2.tar.gz cn-celeb2_v2.tar.gz"
fi

# Stage 0:Download data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "=== Stage 0: Generate metadata ==="
    echo "Download $downloadables_list"
    echo "This may take a long time."
    bash scripts/datasets/download_cnceleb.sh $ROOT_DIR $include_augmented_data
fi

# Stage 1: Decompress data
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Decompress all archives ..."
    echo "This could take some time ..."

    for archive in $downloadables_list; do
        [ ! -f ${ROOT_DIR}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
    done
    [ ! -d "${ROOT_DIR}" ] && mkdir -p "${ROOT_DIR}"

    if [ $include_augmented_data = "true" ] ; then
        if [ ! -d ${ROOT_DIR}/musan ]; then
        tar -xzvf "${ROOT_DIR}/musan.tar.gz" -C "${ROOT_DIR}"
        fi

        if [ ! -d ${ROOT_DIR}/RIRS_NOISES ]; then
        unzip "${ROOT_DIR}/rirs_noises.zip" -d "${ROOT_DIR}"
        fi
    fi

    if [ ! -d "${ROOT_DIR}/${CNCELEB1_FLAC}" ]; then
        tar -xzvf "${ROOT_DIR}/cn-celeb_v2.tar.gz" -C "${ROOT_DIR}"
    fi

    if [ ! -d "${ROOT_DIR}/${CNCELEB2_FLAC}" ]; then
        tar -xzvf "${ROOT_DIR}/cn-celeb2_v2.tar.gz" -C "${ROOT_DIR}"
    fi

    echo "Decompress success !!!"
fi

# Stage 2: Convert FLAC to WAV using sox
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo ""
    echo "=== Stage 2: Convert FLAC to WAV ==="

    echo "🔧 Preparing CNCeleb dataset"
    echo "   Root directory: $ROOT_DIR"
    echo "   Parallel jobs: $num_jobs"

    for subset in $CNCELEB1_FLAC $CNCELEB2_FLAC; do
        PYTHONPATH=$(pwd) python scripts/datasets/flac2wav.py \
            --dataset_dir "$ROOT_DIR/$subset" \
            --nj $num_jobs

        # Copy artifacts from the original flac folders
        if [ "$subset" = "$CNCELEB1_FLAC" ]; then
            cp -r "$ROOT_DIR/$CNCELEB1_FLAC/dev" "$ROOT_DIR/$CNCELEB1_WAV/."
            cp -r "$ROOT_DIR/$CNCELEB1_FLAC/eval/lists" "$ROOT_DIR/$CNCELEB1_WAV/eval/."
        else
            cp "$ROOT_DIR/$CNCELEB2_FLAC/spk.lst" "$ROOT_DIR/$CNCELEB2_WAV/."
        fi
    done
fi

# Stage 3: concatenate short utterances
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ""
    echo "=== Stage 3: Concatenate short utterances ==="
    bash  scripts/datasets/clean_concat_cnceleb.sh "$ROOT_DIR" "$OUTPUT_DIR" "$MAPPING_FILE" 
    bash scripts/datasets/concat_cnceleb.sh "$ROOT_DIR" "$OUTPUT_DIR" "$MAPPING_FILE" "$CNCELEB1_WAV" "$CNCELEB2_WAV" "$TARGET_DURATION" "$MIN_THRESHOLD"
fi


# Stage 4: Run the Python preparation script
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ""
    echo "=== Stage 4: Generate metadata ==="
    PYTHONPATH=$(pwd) python src/datamodules/components/cnceleb/cnceleb_prep.py \
        --root_dir "$ROOT_DIR"
    echo ""
    echo "✅ CNCeleb dataset preparation completed successfully"
fi 
