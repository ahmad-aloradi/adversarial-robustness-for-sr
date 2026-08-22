#!/bin/bash
# Download Tiny-ImageNet-C (Hendrycks & Dietterich, 2019) into
# <data-dir>/tinyimagenet_c/. Idempotent: skips when gaussian_noise/ exists.
# --data-dir must match paths.data_dir; it defaults to the repo's data/.
# Layout: <data-dir>/tinyimagenet_c/{corruption}/{severity}/{wnid}/*.
set -eu

base_dir="data"
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)
            base_dir="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--data-dir DIR]" >&2
            exit 1
            ;;
    esac
done

data_dir="${base_dir}/tinyimagenet_c"
url="https://zenodo.org/records/2536630/files/Tiny-ImageNet-C.tar?download=1"

if [ -d "${data_dir}/gaussian_noise/1" ]; then
    echo "Tiny-ImageNet-C already prepared at ${data_dir}; nothing to do."
    exit 0
fi

mkdir -p "${data_dir}"
tarball="${data_dir}/Tiny-ImageNet-C.tar"
if [ ! -f "${tarball}" ]; then
    echo "Downloading Tiny-ImageNet-C (~7.8 GB) from Zenodo..."
    wget -O "${tarball}.part" "${url}"
    mv "${tarball}.part" "${tarball}"
fi

# The tar extracts to a Tiny-ImageNet-C/ subfolder; flatten it into data_dir.
tar -xf "${tarball}" -C "${data_dir}"
mv "${data_dir}/Tiny-ImageNet-C/"* "${data_dir}/"
rmdir "${data_dir}/Tiny-ImageNet-C"
rm "${tarball}"
echo "Tiny-ImageNet-C ready at ${data_dir}."
