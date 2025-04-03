set -e

# if voxceleb1 and voxceleb2 dir in seprate dirs, define VOX_PATH and uncomment the following lines to create symbolic links to the entire under voxceleb1_2/id*

# VOX_PATH=data/voxceleb
# COMBINED_VOX=voxceleb1_2
# mkdir $VOX_PATH/voxceleb1_2
# for dir in $VOX_PATH/voxceleb1/* $VOX_PATH/voxceleb2/*; do ln -s "$(readlink -f "$dir")" "$COMBINED_VOX/$(basename "$dir")"; done

PYTHONPATH=$(pwd) python src/datamodules/components/voxceleb/voxceleb_prep.py
