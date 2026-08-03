#!/bin/bash
# Download ImageNet-1k (ILSVRC2012) into <data-dir>/imagenet.
# --data-dir must match paths.data_dir (the datamodule reads it as
# ${paths.data_dir}/imagenet); it defaults to the repo's data/.
# Unpacking, MD5 checks and val restructuring are torchvision's official parser.
# Needs ~300 GB peak while the train tar is extracted, ~150 GB after cleanup.
# Layout after this script: <data-dir>/imagenet/{train,val}/{wnid}/*.JPEG.
#
#     bash scripts/datasets/prep_imagenet.sh
#     bash scripts/datasets/prep_imagenet.sh --data-dir /home/woody/.../datasets
set -eu

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${repo_root}/data"
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)
            data_dir="$2"
            shift 2
            ;;
        *)
            echo "usage: $0 [--data-dir DIR]" >&2
            exit 1
            ;;
    esac
done

root="${data_dir}/imagenet"
base_url="https://image-net.org/data/ILSVRC/2012"

mkdir -p "${root}"

count_dirs() { find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l; }
count_tars() { find "$1" -maxdepth 1 -name '*.tar' 2>/dev/null | wc -l; }

# torchvision deletes each inner class tar as it extracts it, so a leftover tar
# (or a short wnid count) means the previous extraction stopped part-way.
extraction_complete() {
    [ "$(count_dirs "${root}/train")" -eq 1000 ] &&
        [ "$(count_tars "${root}/train")" -eq 0 ] &&
        [ "$(count_dirs "${root}/val")" -eq 1000 ]
}

if extraction_complete; then
    echo "ImageNet-1k already extracted at ${root}; verifying."
else
    for archive in ILSVRC2012_devkit_t12.tar.gz ILSVRC2012_img_val.tar ILSVRC2012_img_train.tar; do
        path="${root}/${archive}"
        # -c would resume forever onto an image-net.org login page saved under the .tar name.
        if [ -f "${path}" ] && ! file --brief "${path}" | grep -qE '^(POSIX tar|gzip)'; then
            echo "Discarding non-archive ${path} (image-net.org credentials?)"
            rm -f "${path}"
        fi
        echo "Fetching ${archive} ..."
        # -c resumes a partial file; a 148 GB download rarely survives one session.
        wget -c -P "${root}" "${base_url}/${archive}"
    done
fi

# Extracts whatever is missing (MD5-checked), then asserts the published counts.
# torchvision never resumes a half-extracted split, so that case must fail here.
python -c "
from torchvision.datasets import ImageNet
for split, expected in (('val', 50000), ('train', 1281167)):
    found = len(ImageNet('${root}', split=split))
    assert found == expected, f'{split}: {found} images, expected {expected}. Delete ${root}/{split} and re-run.'
    print(split, found, 'images')
"

rm -f "${root}/ILSVRC2012_img_train.tar" "${root}/ILSVRC2012_img_val.tar"

echo "ImageNet-1k ready at ${root} (1281167 train / 50000 val images)."
