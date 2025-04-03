#!/bin/bash

# ==============================
# VoxCeleb Audio Download and Conversion Script
# (VoxCeleb1 & VoxCeleb2 Dev Audio Only)
# ==============================

read -p "Enter your username: " USERNAME
read -s -p "Enter your password: " PASSWORD
echo ""


# Output directories
VOXCELEB1_DIR="VoxCeleb1"
VOXCELEB2_DIR="VoxCeleb2"

mkdir -p "$VOXCELEB1_DIR"
mkdir -p "$VOXCELEB2_DIR"

# VoxCeleb1 audio archive URL
VOXCELEB1_AUDIO_URL="https://mm.kaist.ac.kr/datasets/voxceleb/vox1_dev_wav.zip"

# VoxCeleb2 audio archive URL
VOXCELEB2_AUDIO_URL="https://mm.kaist.ac.kr/datasets/voxceleb/vox2_dev_aac.tar"

download_and_extract_voxceleb1() {
  cd "$VOXCELEB1_DIR" || exit

  filename=$(basename "$VOXCELEB1_AUDIO_URL")
  if [ -f "$filename" ]; then
    echo "Skipping $filename (already exists)"
  else
    echo "Downloading $filename..."
    wget "$VOXCELEB1_AUDIO_URL"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to download $filename"
      exit 1
    fi
  fi

  echo "Unzipping VoxCeleb1 dev set..."
  unzip -n "$filename" -d vox1_dev_wav
  if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip $filename"
    exit 1
  fi

  cd - > /dev/null
}

download_and_extract_voxceleb2() {
  cd "$VOXCELEB2_DIR" || exit

  filename=$(basename "$VOXCELEB2_AUDIO_URL")
  if [ -f "$filename" ]; then
    echo "Skipping $filename (already exists)"
  else
    echo "Downloading $filename..."
    wget "$VOXCELEB2_AUDIO_URL"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to download $filename"
      exit 1
    fi
  fi

  echo "Extracting VoxCeleb2 dev set..."
  tar xf "$filename"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to extract $filename"
    exit 1
  fi

  cd - > /dev/null
}

convert_aac_to_wav() {
  echo "Converting AAC files to WAV format..."
  find "$VOXCELEB2_DIR/dev/aac" -name "*.m4a" | while read -r file; do
    wav_file="${file%.m4a}.wav"
    if [ ! -f "$wav_file" ]; then
      ffmpeg -loglevel error -y -i "$file" "$wav_file"
      if [ $? -ne 0 ]; then
        echo "Error: Failed to convert $file to WAV"
        exit 1
      fi
    fi
  done
}

echo "=============================="
echo " Downloading and Extracting VoxCeleb1 Audio Files"
echo "=============================="
download_and_extract_voxceleb1

echo "=============================="
echo " Downloading and Extracting VoxCeleb2 Audio Files"
echo "=============================="
download_and_extract_voxceleb2

echo "=============================="
echo " Converting VoxCeleb2 AAC Files to WAV"
echo "=============================="
convert_aac_to_wav

echo "=============================="
echo " DONE! Audio Files Ready."
echo "=============================="
