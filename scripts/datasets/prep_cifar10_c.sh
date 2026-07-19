#!/bin/bash
# Download CIFAR-10-C (Hendrycks & Dietterich, 2019) into data/cifar10_c/.
# Idempotent: skips work when the gaussian_noise.npy + labels.npy pair exists.
# Layout after this script: data/cifar10_c/{corruption}.npy + labels.npy.
set -eu

data_dir="data/cifar10_c"
url="https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"

if [ -f "${data_dir}/gaussian_noise.npy" ] && [ -f "${data_dir}/labels.npy" ]; then
    echo "CIFAR-10-C already prepared at ${data_dir}; nothing to do."
    exit 0
fi

mkdir -p "${data_dir}"
tarball="${data_dir}/CIFAR-10-C.tar"
if [ ! -f "${tarball}" ]; then
    echo "Downloading CIFAR-10-C (~2.9 GB) from Zenodo..."
    wget -O "${tarball}.part" "${url}"
    mv "${tarball}.part" "${tarball}"
fi

# The tar extracts to a CIFAR-10-C/ subfolder; flatten it into data_dir.
tar -xf "${tarball}" -C "${data_dir}"
mv "${data_dir}/CIFAR-10-C/"*.npy "${data_dir}/"
rmdir "${data_dir}/CIFAR-10-C"
rm "${tarball}"
echo "CIFAR-10-C ready at ${data_dir}."
