"""Fabric helpers for launching HPC jobs with clear, human-readable steps.

The tasks in this file perform two high-level chores:

1. Prepare data on the remote cluster so each training job reads from the fast
    local SSD instead of the congested shared filesystem.
2. Generate the SLURM bash scripts that actually start the training runs for
    the VPC (voice privacy challenge) and speaker-verification experiments.
"""

import datetime
import os
import time

from fabric.api import cd, env, run, task
from fabric.contrib.project import rsync_project

# Cluster configuration
env.user = "dsnf101h"  # 'iwal021h'

CLUSTER_NAME = "tinygpu"  # Options: 'tinygpu', 'alex'
env.hosts = (
    ["alex.nhr.fau.de"] if CLUSTER_NAME == "alex" else ["tinyx.nhr.fau.de"]
)
GPU = "a100"  # Options: 'rtx2080ti', 'rtx3080', 'a100'

# Path configuration
PATH_PROJECT = (
    "~/adversarial-robustness-for-sr"  # project folder on the hpc cluster
)
CONDA_ENV = "comfort"

WOODY_DIR = f"/home/woody/{env.user[: 4]}/{env.user}"
VAULT_DIR = f"/home/vault/{env.user[: 4]}/{env.user}"
RESULTS_DIR = os.path.join(VAULT_DIR, "results")
DATA_DIR = os.path.join(WOODY_DIR, "datasets")

# Speaker-verification datasets packaged as fast-to-extract .tar.zst files.
# Paths are relative to DATA_DIR on the cluster.
SV_DATA_ARCHIVES = [
    "cnceleb/CN-Celeb_flac.tar.zst",
    "cnceleb/CN-Celeb2_flac.tar.zst",
    "voxceleb/voxceleb1_2.tar.zst",
]

# sync_results paths - Update these for your specific use case
SYNC_DIR_REMOTE = os.path.join(
    RESULTS_DIR, "train/runs/available_models/*available_models**"
)
SYNC_DIR_LOCAL = "/dataHDD/ahmad/hpc_results/libri_augmented3/"


def timestamp():
    """Return a formatted timestamp string for use in filenames or logs."""
    return (
        str(datetime.datetime.now())
        .split(".")[0]
        .replace(" ", "_")
        .replace(":", "-")
    )


@task
def sync_results():
    """Rsync the entire RESULTS_DIR from the HPC cluster to the local
    machine."""
    rsync_project(
        remote_dir=SYNC_DIR_REMOTE,
        local_dir=SYNC_DIR_LOCAL,
        exclude=[
            ".DS_Store",
            "*.pyc",
            "__pycache__",
            ".git*",
            "*.ckpt",
            "*.pth",
            ".vscode",
        ],
        delete=False,
        upload=False,
    )


def generate_vpc_data_transfer_section(settings):
    """Generate bash code for transferring VPC datasets to local SSD with
    locking.

    Args:
        settings (dict): Dictionary containing data_path, vpc_dirname, and datamodule_dir.

    Returns:
        str: Bash script section for data transfer with race condition protection.
    """
    is_available_models = "available_models" in settings["datamodule_dir"]
    is_libri_included = "LibriSpeech" in settings["datamodule_dir"]

    return f"""
VPC_PATH="{settings['data_path']}/{settings['vpc_dirname']}"
DEST_DIR="$TMPDIR/{settings['vpc_dirname']}"

# Track failed transfers
FAILED_TRANSFERS=()

# Function to transfer and extract model data
transfer_model() {{
    local model=$1

    local tar_file="$VPC_PATH/$model.tar.gz"
    if [[ ! -f "$tar_file" ]]; then
        echo "Error: $tar_file not found" >&2
        FAILED_TRANSFERS+=("$model")
        return 1
    fi

    echo "Transferring $model"
    if ! rsync -ah "$tar_file" "$DEST_DIR/"; then
        echo "Error: Failed to transfer $model" >&2
        FAILED_TRANSFERS+=("$model")
        return 1
    fi

    if ! tar -xzf "$DEST_DIR/$(basename "$tar_file")" -C "$DEST_DIR/"; then
        echo "Error: Failed to extract $model" >&2
        FAILED_TRANSFERS+=("$model")
        return 1
    fi

    # Transfer metadata if needed
    if [[ "$model" != "librispeech" && -d "$VPC_PATH/$model/data/metadata" && ! -d "$DEST_DIR/$model/data/metadata" ]]; then
        if ! rsync -ah "$VPC_PATH/$model/data/metadata" "$DEST_DIR/$model/data/"; then
            echo "Warning: Failed to transfer metadata for $model" >&2
        fi
    fi

    echo "Completed transfer of $model"
    return 0
}}

# Main data transfer with locking mechanism
LOCK_FILE="$TMPDIR/.vpc_data_transfer.lock"

if mkdir "$LOCK_FILE" 2>/dev/null; then
    # This job won the race, it will transfer data
    echo "Starting data transfer (this job acquired the lock)"
    mkdir -p "$DEST_DIR"

    if [[ "{is_available_models}" == "True" ]]; then
        echo "Transferring models in parallel"
        # Process B* and T* models
        for model_tar in "$VPC_PATH"/{{B*,T*}}.tar.gz; do
            if [[ -f "$model_tar" ]]; then
                transfer_model "$(basename "$model_tar" .tar.gz)" &
            fi
        done

        # Transfer librispeech for multiple models
        if [[ "{is_libri_included}" == "True" ]]; then
            transfer_model "librispeech" &
        fi
    else
        # Extract model names from directory
        IFS='_' read -ra MODELS <<< "$(basename "{settings['datamodule_dir']}")"

        for model in "${{MODELS[@]}}"; do
            # Handle special case for LibriSpeech
            if [[ "$model" == "LibriSpeech" ]]; then
                transfer_model "librispeech" &
            else
                transfer_model "$model" &
            fi
        done
    fi

    # Wait for all transfers to complete
    wait

    # Check if any transfers failed
    if [[ ${{#FAILED_TRANSFERS[@]}} -gt 0 ]]; then
        echo "Error: The following transfers failed: ${{FAILED_TRANSFERS[*]}}" >&2
        rmdir "$LOCK_FILE"
        exit 1
    fi

    echo "All data transfers complete successfully"

    # Create a marker file to indicate successful completion
    touch "$DEST_DIR/.transfer_complete"
    rmdir "$LOCK_FILE"
else
    # Another job is transferring or has transferred data
    echo "Another job is handling data transfer, waiting for completion"

    # Wait for the lock to be released and data to be ready
    WAIT_COUNT=0
    MAX_WAIT=180  # 30 minutes (180 * 10 seconds)

    while [[ ! -f "$DEST_DIR/.transfer_complete" && $WAIT_COUNT -lt $MAX_WAIT ]]; do
        sleep 10
        WAIT_COUNT=$((WAIT_COUNT + 1))

        if [[ $((WAIT_COUNT % 6)) -eq 0 ]]; then  # Print every minute
            echo "Still waiting for data transfer... ($((WAIT_COUNT / 6)) minutes)"
        fi
    done

    if [[ ! -f "$DEST_DIR/.transfer_complete" ]]; then
        echo "Error: Timeout waiting for data transfer to complete" >&2
        exit 1
    fi

    echo "Data transfer complete, proceeding with training"
fi

# Verify critical directories exist before proceeding
if [[ ! -d "$DEST_DIR" ]]; then
    echo "Error: Data directory $DEST_DIR does not exist" >&2
    exit 1
fi
"""


def create_vpc_bash_script(settings, script_arguments):
    """Creates a SLURM bash script for VPC training jobs.

    Args:
        settings (dict): Dictionary containing job configuration settings.
        script_arguments (dict): Dictionary of arguments to pass to the training script.

    Returns:
        str: A bash script as a string, ready to be submitted to SLURM.
    """
    script_arguments_str = " ".join(
        [f"{k}={v}" for k, v in script_arguments.items()]
    )
    data_setup_section = generate_vpc_data_transfer_section(settings)

    job_string = f"""#!/bin/bash -l
#SBATCH --job-name={settings['job_name']}
#SBATCH --clusters={settings['cluster']}
#SBATCH --partition={settings['gpu']}
#SBATCH --nodes={settings['num_nodes']}
#SBATCH --gres=gpu:{settings['gpu']}:{settings['num_gpus']}
#SBATCH --time={settings['walltime']}
#SBATCH --export=NONE

# Load required modules and activate environment
module load cuda/{settings['cuda']}
source ~/miniconda3/bin/activate {settings['env_name']}
cd {settings['path_project']}

{data_setup_section}

# Start training
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

echo 'Starting training'
python {settings['script_name']} {script_arguments_str} paths.data_dir=$TMPDIR
"""
    return job_string


def generate_sv_data_transfer_section(settings):
    """Generate bash code for transferring SV datasets to local SSD.

    Args:
        settings (dict): Dictionary containing data_path and sv_archives configuration.

    Returns:
        str: Bash script section for data transfer with error handling.
    """
    archives = settings.get("sv_archives", SV_DATA_ARCHIVES)
    archives_literal = " ".join([f'"{a}"' for a in archives])

    return f"""
DATA_ROOT="{settings['data_path']}"
DEST_DATA_ROOT="$TMPDIR/datasets"
DATA_ARCHIVES=({archives_literal})

# Track failed transfers
declare -a FAILED_TRANSFERS

# Helper: copy compressed datasets to the node-local SSD (automatically deleted
# after the job ends).
stage_archive() {{
    local relative="$1"
    local archive_path="$DATA_ROOT/$relative"
    local archive_name=$(basename "$relative")
    local extract_dir="$archive_name"
    extract_dir="${{extract_dir%.tar.gz}}"
    extract_dir="${{extract_dir%.tar.zst}}"
    local target_dir="$DEST_DATA_ROOT/$(dirname "$relative")"

    if [[ ! -f "$archive_path" ]]; then
        echo "Error: Archive $archive_path not found" >&2
        FAILED_TRANSFERS+=("$relative")
        return 1
    fi

    mkdir -p "$target_dir"

    if [[ -d "$target_dir/$extract_dir" ]]; then
        echo "$relative already extracted"
        return 0
    fi

    echo "Transferring $relative -> $target_dir"
    if ! rsync -ah "$archive_path" "$target_dir/"; then
        echo "Error: Failed to transfer $relative" >&2
        FAILED_TRANSFERS+=("$relative")
        return 1
    fi

    local tar_file="$target_dir/$archive_name"
    local tar_cmd=(tar)

    if [[ "$archive_name" == *.tar.zst ]]; then
        local zstd_prog=""
        if command -v zstdmt >/dev/null 2>&1; then
            zstd_prog=$(command -v zstdmt)
        elif command -v zstd >/dev/null 2>&1; then
            zstd_prog=$(command -v zstd)
        fi

        if [[ -z "$zstd_prog" ]]; then
            echo "Error: zstd/zstdmt not found for $archive_name" >&2
            FAILED_TRANSFERS+=("$relative")
            return 1
        fi

        tar_cmd+=(--use-compress-program "$zstd_prog" -xf)
    elif [[ "$archive_name" == *.tar.gz ]]; then
        if command -v pigz >/dev/null 2>&1; then
            tar_cmd+=(--use-compress-program "$(command -v pigz)" -xf)
        else
            tar_cmd+=(-xzf)
        fi
    else
        echo "Error: Unsupported archive format $archive_name" >&2
        FAILED_TRANSFERS+=("$relative")
        return 1
    fi

    echo "Extracting $archive_name"
    if ! "${{tar_cmd[@]}}" "$tar_file" -C "$target_dir"; then
        echo "Error: Failed to extract $archive_name" >&2
        FAILED_TRANSFERS+=("$relative")
        return 1
    fi

    return 0
}}

# Transfer metadata folders
transfer_metadata() {{
    local dataset_type="$1"  # "voxceleb" or "cnceleb"
    local metadata_src=""
    local metadata_dest=""

    if [[ "$dataset_type" == "voxceleb" ]]; then
        metadata_src="$DATA_ROOT/voxceleb/voxceleb_metadata"
        metadata_dest="$DEST_DATA_ROOT/voxceleb/voxceleb_metadata"
    elif [[ "$dataset_type" == "cnceleb" ]]; then
        metadata_src="$DATA_ROOT/cnceleb/metadata"
        metadata_dest="$DEST_DATA_ROOT/cnceleb/metadata"
    fi

    if [[ -d "$metadata_src" && ! -e "$metadata_dest" ]]; then
        echo "Transferring metadata for $dataset_type"
        mkdir -p "$(dirname "$metadata_dest")"
        if ! rsync -ah "$metadata_src" "$(dirname "$metadata_dest")/"; then
            echo "Warning: Failed to transfer metadata for $dataset_type" >&2
        fi
    fi
}}

echo "Staging speaker verification datasets to $DEST_DATA_ROOT"
mkdir -p "$DEST_DATA_ROOT"
stage_start=$(date +%s)

for archive in "${{DATA_ARCHIVES[@]}}"; do
    stage_archive "$archive" &
done
wait

# Check for failed transfers
if [[ ${{#FAILED_TRANSFERS[@]}} -gt 0 ]]; then
    echo "Error: The following dataset transfers failed:" >&2
    printf '%s\\n' "${{FAILED_TRANSFERS[@]}}" >&2
    exit 1
fi

# Transfer metadata for each dataset type
transfer_metadata "voxceleb"
transfer_metadata "cnceleb"

stage_end=$(date +%s)
stage_duration=$((stage_end - stage_start))
echo "Dataset staging complete in $((stage_duration / 60)) minutes $((stage_duration % 60)) seconds"
"""


def create_sv_bash_script(settings, script_arguments, transfer_data=False):
    """Creates a SLURM bash script for SV training jobs.

    Args:
        settings (dict): Dictionary containing job configuration settings.
        script_arguments (dict): Dictionary of arguments to pass to the training script.
        transfer_data (bool): If True, transfer data to local SSD. If False, use shared filesystem.

    Returns:
        str: A bash script as a string, ready to be submitted to SLURM.
    """
    script_arguments_str = " ".join(
        [f"{k}={v}" for k, v in script_arguments.items()]
    )

    # Determine data directory and transfer section based on transfer_data flag
    if transfer_data:
        data_dir_path = "$DEST_DATA_ROOT"
        data_setup_section = generate_sv_data_transfer_section(settings)
    else:
        data_dir_path = f'"{settings["data_path"]}"'
        data_setup_section = f'echo "Using shared filesystem at {settings["data_path"]} (no data transfer)"'

    job_string = f"""#!/bin/bash -l
#SBATCH --job-name={settings['job_name']}
#SBATCH --clusters={settings['cluster']}
#SBATCH --partition={settings['gpu']}
#SBATCH --nodes={settings['num_nodes']}
#SBATCH --gres=gpu:{settings['gpu']}:{settings['num_gpus']}
#SBATCH --time={settings['walltime']}
#SBATCH --export=NONE

# Load required modules and activate environment
module load cuda/{settings['cuda']}
source ~/miniconda3/bin/activate {settings['env_name']}
cd {settings['path_project']}

{data_setup_section}

# Start the training process
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

echo 'Starting training'
python {settings['script_name']} {script_arguments_str} paths.data_dir={data_dir_path}
"""
    return job_string


def run_bash_script(pbs):
    """Submit a bash script to SLURM."""
    with cd(PATH_PROJECT):
        run(
            f'PATH=$PATH:/apps/slurm/current/bin && cat << "EOF" |  sbatch\n{pbs}\nEOF',
            shell=True,
        )


@task
def scancel():
    """Stop all jobs associated to your username."""
    run("squeue | grep $USER | tr -s ' ' | cut -d ' ' -f 2 | xargs scancel")


@task
def squeue():
    """Get the queue status of the cluster."""
    run("squeue -l")


@task
def clean_logs():
    """Delete log files."""
    with cd(PATH_PROJECT):
        run("rm -f *.o* *.e*")


@task
def cmd(command):
    """Execute an arbitrary command on the remote cluster."""
    run(command)


@task
def lsp():
    """Returns the names of pending jobs on the queue."""
    print(run('squeue -t PENDING --format "%.j" -h').split())


@task
def lsr():
    """Returns the names of running jobs."""
    print(run('squeue -t RUNNING --format "%.j" -h').split())


@task
def lsa():
    """Returns the names of all jobs."""
    print(run('squeue --format "%.j" -h').split())


def check_running_pending(job_name):
    """Check if a job is pending or running.

    Args:
        job_name (str): Name of the job to check

    Returns:
        bool: True if the job is pending or running, False otherwise
    """
    result = run('squeue -t PENDING,RUNNING --format "%.j" -h', quiet=True)
    files = result.split() if result else []
    return job_name in files


def check_file_exists(file_path):
    """Check if a file exists on the remote cluster.

    Args:
        file_path (str): Path to the file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    result = run(f"test -f {file_path} && echo 1 || echo 0", quiet=True)
    return result.strip() == "1"


def check_test_artifacts_complete(run_dir):
    """Return True if the run directory contains completed test artifacts.

    A run is considered "already tested" when:
    - {run_dir}/test_artifacts exists, and
    - it contains at least one non-hidden immediate subdirectory, and
    - every non-hidden immediate subdirectory contains a file named COMPLETE.

    Hidden directories (starting with _ or .) are skipped (e.g., _cohort_cache).

    This matches the on-cluster layout like:
        test_artifacts/<test_set>/{TIMESTAMP,cache,COMPLETE,LAST_RUN}
    """
    test_root = os.path.join(run_dir, "test_artifacts")

    # Use a small shell script to avoid multiple round-trips.
    cmd = f"""
test_root="{test_root}"

if [[ ! -d "$test_root" ]]; then
    echo 0
    exit 0
fi

# Gather non-hidden immediate subdirectories under test_artifacts.
# Skip directories starting with _ or . (e.g., _cohort_cache)
mapfile -t subdirs < <(find "$test_root" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | while read -r d; do
    basename_d=$(basename "$d")
    if [[ ! "$basename_d" =~ ^[._] ]]; then
        echo "$d"
    fi
done)

if [[ ${{#subdirs[@]}} -eq 0 ]]; then
    echo 0
    exit 0
fi

for d in "${{subdirs[@]}}"; do
    if [[ ! -f "$d/COMPLETE" ]]; then
        echo 0
        exit 0
    fi
done

echo 1
"""

    result = run(cmd, quiet=True)
    return result.strip() == "1"


@task
def run_vpc():
    """Generate and submit all VPC training jobs."""

    BATCH_SIZE = 32
    NUM_AUG = 1
    max_epochs = 25
    max_duration = 10
    dataset_dirname = "vpc2025_official"

    settings = {
        "script_name": "src/train.py",
        "cluster": CLUSTER_NAME,
        "path_project": PATH_PROJECT,
        "env_name": CONDA_ENV,
        "data_path": DATA_DIR,
        "vpc_dirname": dataset_dirname,
        "gpu": GPU,
        "num_nodes": 1,
        "walltime": "24:00:00",
        "num_gpus": 1,
        "cuda": "11.8.0",
    }

    datasets = [
        "{B3: ${datamodule.available_models.B3}}",
        "{B4: ${datamodule.available_models.B4}}",
        "{B5: ${datamodule.available_models.B5}}",
        "{T8-5: ${datamodule.available_models.T8-5}}",
        "{T10-2: ${datamodule.available_models.T10-2}}",
        "{T12-5: ${datamodule.available_models.T12-5}}",
        "{T25-1: ${datamodule.available_models.T25-1}}",
    ]
    experiments = ["vpc_amm_cyclic_audio_from_scratch"]

    for experiment in experiments:
        for dataset in datasets:

            # Parse dataset string
            if dataset.startswith("{") and ":" in dataset:
                if "," in dataset:
                    dataset_name = "_".join(
                        item.split(":")[0].strip("{").strip()
                        for item in dataset.split(",")
                    )
                else:
                    dataset_name = dataset.split(":")[0].strip("{").strip()
            else:
                dataset_name = "available_models"

            job_name = f"{experiment}-{dataset_name}-max_dur{max_duration}-bs{BATCH_SIZE}"

            settings["job_name"] = job_name
            settings["datamodule_dir"] = (
                dataset_dirname + os.sep + dataset_name
            )
            name = dataset_name + os.sep + job_name

            script_arguments = {
                "datamodule": "datasets/vpc",
                "experiment": f"vpc/{experiment}",
                "module": "vpc",
                "trainer": "gpu",
                "name": name,
                "datamodule.models": f"'{dataset}'",
                "datamodule.loaders.train.batch_size": BATCH_SIZE
                if "aug" not in experiment
                else BATCH_SIZE // NUM_AUG,
                "datamodule.loaders.valid.batch_size": BATCH_SIZE
                if "aug" not in experiment
                else BATCH_SIZE // NUM_AUG,
                "datamodule.dataset.max_duration": max_duration,
                "trainer.max_epochs": max_epochs,
                "paths.log_dir": f"{RESULTS_DIR}",
                "hydra.run.dir": f"{RESULTS_DIR}/train/runs/{name}",
                "trainer.num_sanity_val_steps": 0,
            }

            # Skip if job is already running or pending
            if check_running_pending(job_name):
                print(f"Skipping {job_name} - already running or pending")
                continue

            # Check for existing checkpoint
            last_ckpt = os.path.join(
                script_arguments["hydra.run.dir"], "checkpoints/last.ckpt"
            )
            if check_file_exists(last_ckpt):
                script_arguments["ckpt_path"] = last_ckpt
                print(f"Resuming {job_name} from checkpoint")

            bash_script = create_vpc_bash_script(settings, script_arguments)
            run_bash_script(bash_script)
            print(f"Submitted job: {job_name}")
            time.sleep(1.0)


@task
def run_sv(transfer_data="false"):
    """Generate and submit all SV training jobs.

    Args:
        transfer_data (str): 'true' to transfer data to local SSD, 'false' to use shared filesystem
    """
    
    assert CLUSTER_NAME == "alex", "Run exps on Alex to avoid overloading the shared filesystem with multiple simultaneous transfers"
    transfer_data_bool = transfer_data.lower() in ("true", "1", "yes")

    BATCH_SIZE = 128
    apply_vad = False
    schedule_type = "constant"  # Options: 'constant', 'linear'
    num_ckpt_avg = 3  # Number of checkpoints to average for final model (only applicable for certain experiments)

    settings = {
        "script_name": "src/train.py",
        "cluster": CLUSTER_NAME,
        "path_project": PATH_PROJECT,
        "env_name": CONDA_ENV,
        "data_path": DATA_DIR,
        "gpu": GPU,
        "num_nodes": 1,
        "walltime": "24:00:00",
        "num_gpus": 1,
        "cuda": "12.9.0",
    }

    experiments = [
        # #Bregman experiments
        'sv_bregman_adabreg_warmup',
        'sv_bregman_adabreg',
        'sv_bregman_linbreg_warmup',
        'sv_bregman_linbreg',
        # #Pruning experiments
        'sv_pruning_mag_struct_onetime',
        'sv_pruning_mag_unstruct_onetime',
        'sv_pruning_mag_struct',
        'sv_pruning_mag_unstruct',
        # #Baselines
        'sv_wespeaker',
        # 'sv_vanilla',
    ]

    ramp_up_experiments = [
        "sv_pruning_mag_struct",
        "sv_pruning_mag_unstruct",
    ]

    # Get SV models from config directory
    config_dir = "../configs/module/sv_model"
    assert os.path.exists(config_dir), f"Config directory {config_dir} does not exist"
    sv_models = ["wespeaker_ecapa_tdnn", 
                 "wespeaker_pretrained_ecapa_tdnn",
                 "wespeaker_resnet34",
                 "wespeaker_pretrained_resnet34",
                 "wespeaker_resnet152",
                 "wespeaker_pretrained_resnet152"
                 ]
    
    # dataset_names = ["datasets/voxceleb", "datasets/cnceleb", "multi_sv"]
    dataset_names = ["multi_sv"]

    for experiment in experiments:
        for sv_model in sv_models:
            is_pretrained = "pretrained" in sv_model
            is_onetime = "onetime" in experiment

            # skip when model type and experiment type disagree
            if is_pretrained != is_onetime:
                if is_pretrained:
                    print(f"\nSkipping {experiment} with {sv_model}  (Should be trained from scratch only)!\n")
                else:
                    print(f"\nSkipping {experiment} with {sv_model}  (onetime pruning is for fine-tuning)!\n")
                continue

            for dataset_name in dataset_names:
                virtual_spks = "False"

                #  Handling epochs
                ############################################################
                epochs_to_ramp = None
                if dataset_name == "datasets/cnceleb":
                    max_epochs = 40
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                elif dataset_name == "datasets/voxceleb":
                    max_epochs = 20
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                elif dataset_name == "multi_sv":
                    max_epochs = 20
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                else:
                    raise ValueError(
                        f"Unexpected dataset name: {dataset_name}"
                    )
                ############################################################

                # Handling of special cases:
                # 1. batch_size of 32 does not fit for resnet293 on the A100! --> use 16
                batch_size = BATCH_SIZE if "resnet293" not in sv_model else 16
                # 2. cannot use pruning with certain models
                if "pruning" in experiment:
                    if (
                        sv_model == "speechbrain_pretrained_ecapa_tdnn"
                        or sv_model
                        == "pretraiend_ecapa2"  # serialized implementation
                    ):
                        print(
                            f"\nSkipping {experiment} with {sv_model} - pruning not supported for this model!!\n"
                        )
                        continue

                # shared settings
                ############################################################
                if epochs_to_ramp:
                    job_name = f"{experiment}-ramp{epochs_to_ramp}_{schedule_type}-{sv_model}-{os.path.basename(dataset_name)}-virtual_spks-{virtual_spks}-bs{batch_size}-ckpt_avg{num_ckpt_avg}-vad{apply_vad}-max_epochs{max_epochs}"
                else:
                    job_name = f"{experiment}-{sv_model}-{os.path.basename(dataset_name)}-virtual_spks-{virtual_spks}-bs{batch_size}-vad{apply_vad}-ckpt_avg{num_ckpt_avg}-max_epochs{max_epochs}"

                settings["job_name"] = job_name
                name = os.path.basename(dataset_name) + os.sep + job_name

                script_arguments = {
                    "datamodule": dataset_name,
                    "experiment": f"sv/{experiment}",
                    "module": "sv",
                    "trainer": "gpu",
                    "name": name,
                    "module/sv_model": sv_model,
                    "logger": "many_loggers.yaml",
                    "datamodule.loaders.train.batch_size": batch_size,
                    "datamodule.loaders.valid.batch_size": batch_size,
                    "trainer.max_epochs": max_epochs,
                    "paths.log_dir": f"{RESULTS_DIR}",
                    "hydra.run.dir": f"{RESULTS_DIR}/train/runs/{name}",
                    "trainer.num_sanity_val_steps": 0,
                    "datamodule.dataset.vad.enabled": str(apply_vad),
                    "callbacks.checkpoint_averaging.num_checkpoints": num_ckpt_avg,
                }

                ############################################################
                # Handle virtual speakers setting for CN-Celeb vs VoxCeleb
                script_arguments.update(
                    {
                        "module.data_augmentation.augmentations.wav_augmenter.speed_perturb.virtual_speakers": virtual_spks,
                    }
                )

                # handle epochs to ramp
                if epochs_to_ramp:
                    epochs_to_ramp_key = "callbacks.model_pruning.epochs_to_ramp"
                    script_arguments.update(
                        {epochs_to_ramp_key: epochs_to_ramp}
                    )
                    script_arguments.update(
                        {
                            "callbacks.model_pruning.schedule_type": schedule_type
                        }
                    )

                # Jobs submission logic (skipping already running/pending jobs, resuming from checkpoint, skipping if testing is already complete)
                ############################################################
                # Skip if job is already running or pending
                if check_running_pending(job_name):
                    print(f"Skipping {job_name} - already running or pending")
                    continue

                # Skip job if the testing phase is done
                if check_test_artifacts_complete(
                    script_arguments["hydra.run.dir"]
                ):
                    print(f"Skipping {job_name} - testing already complete")
                    continue

                # Check for existing checkpoint
                last_ckpt = os.path.join(
                    script_arguments["hydra.run.dir"], "checkpoints/last.ckpt"
                )
                if check_file_exists(last_ckpt):
                    script_arguments["ckpt_path"] = last_ckpt
                    print(f"Resuming {job_name} from checkpoint")

                bash_script = create_sv_bash_script(
                    settings,
                    script_arguments,
                    transfer_data=transfer_data_bool,
                )
                run_bash_script(bash_script)

                print(
                    f"Submitted job: {job_name} (transfer_data={transfer_data_bool})"
                )
                time.sleep(0.1)



@task
def run_baselines(transfer_data="false"):
    """Generate and submit all SV training jobs.

    Args:
        transfer_data (str): 'true' to transfer data to local SSD, 'false' to use shared filesystem
    """
    assert CLUSTER_NAME == "tinygpu", "Run exps on TinyGPU to avoid overloading the shared filesystem with multiple simultaneous transfers"
    transfer_data_bool = transfer_data.lower() in ("true", "1", "yes")

    BATCH_SIZE = 128
    apply_vad = False
    schedule_type = "constant"  # Options: 'constant', 'linear'
    num_ckpt_avg = 3  # Number of checkpoints to average for final model (only applicable for certain experiments)

    settings = {
        "script_name": "src/train.py",
        "cluster": CLUSTER_NAME,
        "path_project": PATH_PROJECT,
        "env_name": CONDA_ENV,
        "data_path": DATA_DIR,
        "gpu": GPU,
        "num_nodes": 1,
        "walltime": "24:00:00",
        "num_gpus": 1,
        "cuda": "12.9.0",
    }

    experiments = [
        # #Bregman experiments
        'sv_bregman_adabreg_warmup',
        'sv_bregman_adabreg',
        'sv_bregman_linbreg_warmup',
        'sv_bregman_linbreg',
        # #Pruning experiments
        'sv_pruning_mag_struct_onetime',
        'sv_pruning_mag_unstruct_onetime',
        'sv_pruning_mag_struct',
        'sv_pruning_mag_unstruct',
        # #Baselines
        'sv_wespeaker',
        # 'sv_vanilla',
    ]

    ramp_up_experiments = [
        "sv_pruning_mag_struct",
        "sv_pruning_mag_unstruct",
    ]

    # Get SV models from config directory
    config_dir = "../configs/module/sv_model"
    assert os.path.exists(config_dir), f"Config directory {config_dir} does not exist"
    sv_models = ["wespeaker_ecapa_tdnn", 
                #  "wespeaker_pretrained_ecapa_tdnn",
                 "wespeaker_resnet34",
                #  "wespeaker_pretrained_resnet34",
                 "wespeaker_resnet152",
                #  "wespeaker_pretrained_resnet152"
                 ]
    
    dataset_names = ["datasets/voxceleb", "datasets/cnceleb"]

    for experiment in experiments:
        for sv_model in sv_models:
            is_pretrained = "pretrained" in sv_model
            is_onetime = "onetime" in experiment

            # skip when model type and experiment type disagree
            if is_pretrained != is_onetime:
                if is_pretrained:
                    print(f"\nSkipping {experiment} with {sv_model}  (Should be trained from scratch only)!\n")
                else:
                    print(f"\nSkipping {experiment} with {sv_model}  (onetime pruning is for fine-tuning)!\n")
                continue

            for dataset_name in dataset_names:
                virtual_spks = "False"

                #  Handling epochs
                ############################################################
                epochs_to_ramp = None
                if dataset_name == "datasets/cnceleb":
                    max_epochs = 40
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                elif dataset_name == "datasets/voxceleb":
                    max_epochs = 10
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                elif dataset_name == "multi_sv":
                    max_epochs = 10
                    if experiment in ramp_up_experiments:
                        epochs_to_ramp = 10
                        max_epochs += epochs_to_ramp
                else:
                    raise ValueError(
                        f"Unexpected dataset name: {dataset_name}"
                    )
                ############################################################

                # Handling of special cases:
                # 1. batch_size of 32 does not fit for resnet293 on the A100! --> use 16
                batch_size = BATCH_SIZE if "resnet293" not in sv_model else 16
                # 2. cannot use pruning with certain models
                if "pruning" in experiment:
                    if (
                        sv_model == "speechbrain_pretrained_ecapa_tdnn"
                        or sv_model
                        == "pretraiend_ecapa2"  # serialized implementation
                    ):
                        print(
                            f"\nSkipping {experiment} with {sv_model} - pruning not supported for this model!!\n"
                        )
                        continue

                # shared settings
                ############################################################
                if epochs_to_ramp:
                    job_name = f"{experiment}-ramp{epochs_to_ramp}_{schedule_type}-{sv_model}-{os.path.basename(dataset_name)}-virtual_spks-{virtual_spks}-bs{batch_size}-ckpt_avg{num_ckpt_avg}-vad{apply_vad}-max_epochs{max_epochs}"
                else:
                    job_name = f"{experiment}-{sv_model}-{os.path.basename(dataset_name)}-virtual_spks-{virtual_spks}-bs{batch_size}-vad{apply_vad}-ckpt_avg{num_ckpt_avg}-max_epochs{max_epochs}"

                settings["job_name"] = job_name
                name = os.path.basename(dataset_name) + os.sep + job_name

                script_arguments = {
                    "datamodule": dataset_name,
                    "experiment": f"sv/{experiment}",
                    "module": "sv",
                    "trainer": "gpu",
                    "name": name,
                    "module/sv_model": sv_model,
                    "logger": "many_loggers.yaml",
                    "datamodule.loaders.train.batch_size": batch_size,
                    "datamodule.loaders.valid.batch_size": batch_size,
                    "trainer.max_epochs": max_epochs,
                    "paths.log_dir": f"{RESULTS_DIR}",
                    "hydra.run.dir": f"{RESULTS_DIR}/train/runs/{name}",
                    "trainer.num_sanity_val_steps": 0,
                    "datamodule.dataset.vad.enabled": str(apply_vad),
                    "callbacks.checkpoint_averaging.num_checkpoints": num_ckpt_avg,
                }

                ############################################################
                # Handle virtual speakers setting for CN-Celeb vs VoxCeleb
                script_arguments.update(
                    {
                        "module.data_augmentation.augmentations.wav_augmenter.speed_perturb.virtual_speakers": virtual_spks,
                    }
                )

                # handle epochs to ramp
                if epochs_to_ramp:
                    epochs_to_ramp_key = "callbacks.model_pruning.epochs_to_ramp"
                    script_arguments.update(
                        {epochs_to_ramp_key: epochs_to_ramp}
                    )
                    script_arguments.update(
                        {
                            "callbacks.model_pruning.schedule_type": schedule_type
                        }
                    )

                # Jobs submission logic (skipping already running/pending jobs, resuming from checkpoint, skipping if testing is already complete)
                ############################################################
                # Skip if job is already running or pending
                if check_running_pending(job_name):
                    print(f"Skipping {job_name} - already running or pending")
                    continue

                # Skip job if the testing phase is done
                if check_test_artifacts_complete(
                    script_arguments["hydra.run.dir"]
                ):
                    print(f"Skipping {job_name} - testing already complete")
                    continue

                # Check for existing checkpoint
                last_ckpt = os.path.join(
                    script_arguments["hydra.run.dir"], "checkpoints/last.ckpt"
                )
                if check_file_exists(last_ckpt):
                    script_arguments["ckpt_path"] = last_ckpt
                    print(f"Resuming {job_name} from checkpoint")

                bash_script = create_sv_bash_script(
                    settings,
                    script_arguments,
                    transfer_data=transfer_data_bool,
                )
                run_bash_script(bash_script)

                print(
                    f"Submitted job: {job_name} (transfer_data={transfer_data_bool})"
                )
                time.sleep(0.1)
