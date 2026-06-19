"""Fabric helpers for launching speaker-verification jobs on the HPC cluster.

The tasks in this file perform two high-level chores:

1. Stage the SV datasets onto each node's fast local SSD so a training job
    reads from local disk instead of the congested shared filesystem.
2. Generate and submit the SLURM bash scripts that start the training and
    evaluation runs.
"""

import datetime
import os
import time

from fabric.api import cd, env, local, run, task
from fabric.contrib.project import rsync_project

# Cluster configuration
env.user = "dsnf101h"  # 'iwal021h'

CLUSTER_NAME = "alex"  # Options: 'tinygpu', 'alex'
env.hosts = (
    ["alex.nhr.fau.de"] if CLUSTER_NAME == "alex" else ["tinyx.nhr.fau.de"]
)
GPU = "a40"  # Options: 'rtx2080ti', 'rtx3080', 'a100'

# Path configuration
PATH_PROJECT = (
    "~/adversarial-robustness-for-sr"  # project folder on the hpc cluster
)
CONDA_ENV = "comfort"

HPC_HOME_DIR = f"/home/hpc/{env.user[: 4]}/{env.user}"
WOODY_DIR = f"/home/woody/{env.user[: 4]}/{env.user}"
VAULT_DIR = f"/home/vault/{env.user[: 4]}/{env.user}"
RESULTS_DIR = os.path.join(VAULT_DIR, "results")
DATA_DIR = os.path.join(WOODY_DIR, "datasets")

# Speaker-verification datasets, shipped as .tar.gz archives under DATA_DIR.
# Each dataset names its wav archives plus the metadata folder that ships
# alongside them (not inside the archives). Paths are relative to DATA_DIR.
SV_DATASETS = {
    "voxceleb": {
        "archives": ["voxceleb/voxceleb1_2.tar.gz"],
        "metadata": "voxceleb/voxceleb_metadata",
    },
    "cnceleb": {
        "archives": [
            "cnceleb/CN-Celeb_wav.tar.gz",
            "cnceleb/CN-Celeb2_wav.tar.gz",
            "cnceleb/concatenated.tar.gz",
        ],
        "metadata": "cnceleb/metadata",
    },
}

# sync_results paths - Update these for your specific use case.
dataset = "cnceleb"  # "cnceleb" or "multi_sv"
SYNC_DIR_REMOTE = os.path.join(RESULTS_DIR, f"train/runs/{dataset}/*")
SYNC_DIR_LOCAL = f"/dataHDD/ahmad/comfort26_sem/{dataset}"


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
            "wandb",
        ],
        delete=False,
        upload=False,
    )


@task
def sync_tb():
    """Rsync only TensorBoard data from the HPC cluster to the local
    machine."""
    rsync_project(
        remote_dir=SYNC_DIR_REMOTE,
        local_dir=SYNC_DIR_LOCAL,
        # We drop the 'exclude' parameter to ensure strict rule ordering via extra_opts
        extra_opts='-m --include="*/" --include="events.out.tfevents.*" --exclude="*"',
        delete=False,
        upload=False,
    )


@task
def sync_results_for_test():
    """Recursively rsync specific artifacts, checkpoints, and logs from the
    HPC."""
    rsync_project(
        remote_dir=SYNC_DIR_REMOTE,
        local_dir=SYNC_DIR_LOCAL,
        extra_opts=(
            "-m "  # prune empty directories
            "--include='*/' "  # allow traversing directories
            "--include='**/*artifacts/**' "  # include any artifacts dir + contents
            "--include='**/checkpoints/**' "  # include any checkpoints dir + contents
            "--include='**/train_log.txt' "  # include train_log.txt anywhere
            "--exclude='*'"  # exclude everything else
        ),
        delete=False,
        upload=False,
    )


def generate_sv_data_transfer_section(settings):
    """Generate the bash that stages SV datasets onto the node-local SSD.

    Datasets ship as ``.tar.gz`` archives under ``DATA_DIR`` on the cluster.
    The datamodule selects which datasets to stage; their wav archives, the
    matching metadata folder, and the shared RIRS/Noise corpus are copied to
    ``$TMPDIR/datasets`` and extracted there. Staging runs one item at a time
    and aborts the job on the first failure, so a half-populated SSD never
    reaches the training step.

    Args:
        settings (dict): Must contain ``data_path`` (the cluster ``DATA_DIR``)
            and ``datamodule`` (the Hydra datamodule name, e.g. ``multi_sv``).

    Returns:
        str: Bash to drop into the SLURM script before the training command.
    """
    datamodule = str(settings["datamodule"]).lower()
    if "multi_sv" in datamodule:
        datasets = ["voxceleb", "cnceleb"]
    elif "voxceleb" in datamodule:
        datasets = ["voxceleb"]
    elif "cnceleb" in datamodule:
        datasets = ["cnceleb"]
    else:
        raise ValueError(
            f"Cannot map datamodule '{settings['datamodule']}' to SV datasets; "
            f"known datasets: {sorted(SV_DATASETS)}"
        )

    archives = [
        archive
        for name in datasets
        for archive in SV_DATASETS[name]["archives"]
    ]
    metadata_dirs = [SV_DATASETS[name]["metadata"] for name in datasets]

    archives_literal = " ".join(f'"{a}"' for a in archives)
    metadata_literal = " ".join(f'"{m}"' for m in metadata_dirs)

    return f"""
DATA_ROOT="{settings['data_path']}"
DEST_DATA_ROOT="$TMPDIR/datasets"
ARCHIVES=({archives_literal})
METADATA_DIRS=({metadata_literal})

# Copy a .tar.gz archive to the node-local SSD and extract it in place.
stage_archive() {{
    local relative="$1"
    local src="$DATA_ROOT/$relative"
    local dest_dir="$DEST_DATA_ROOT/$(dirname "$relative")"

    if [[ ! -f "$src" ]]; then
        echo "Error: archive $src not found" >&2
        exit 1
    fi

    mkdir -p "$dest_dir"
    echo "Transferring $relative -> $dest_dir"
    rsync -ah "$src" "$dest_dir/" || {{
        echo "Error: failed to copy $relative" >&2
        exit 1
    }}

    echo "Extracting $(basename "$relative")"
    tar -xzf "$dest_dir/$(basename "$relative")" -C "$dest_dir" || {{
        echo "Error: failed to extract $relative" >&2
        exit 1
    }}
}}

# Copy an auxiliary directory (metadata, RIRS/Noise) to the SSD verbatim.
stage_dir() {{
    local relative="$1"
    local src="$DATA_ROOT/$relative"
    local dest_dir="$DEST_DATA_ROOT/$(dirname "$relative")"

    if [[ ! -d "$src" ]]; then
        echo "Error: directory $src not found" >&2
        exit 1
    fi

    mkdir -p "$dest_dir"
    echo "Transferring $relative -> $dest_dir"
    rsync -ah "$src" "$dest_dir/" || {{
        echo "Error: failed to copy $relative" >&2
        exit 1
    }}
}}

echo "Staging speaker verification datasets to $DEST_DATA_ROOT"
mkdir -p "$DEST_DATA_ROOT"
stage_start=$(date +%s)

for archive in "${{ARCHIVES[@]}}"; do
    stage_archive "$archive"
done

for metadata in "${{METADATA_DIRS[@]}}"; do
    stage_dir "$metadata"
done

stage_dir "RIRS_NOISES"

stage_end=$(date +%s)
elapsed=$((stage_end - stage_start))
echo "Dataset staging complete in $((elapsed / 60))m $((elapsed % 60))s"
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
        settings = settings.copy()
        settings["datamodule"] = script_arguments["datamodule"]
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
            shell=True,  # nosec B604
        )


@task
def clean_timelimit_logs():
    """Delete *.o* log files whose job was killed by the SLURM time limit.

    Scans every *.o* file in the project root on the remote cluster and removes
    those that contain the SLURM termination marker 'DUE TO TIME LIMIT ***'.
    """
    with cd(PATH_PROJECT):
        run("bash scripts/cleanup_timelimit_logs.sh .")


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
    """Check if a job is pending or running on any cluster.

    Runs from the local machine, which has SSH access to both
    cluster login nodes.  Each cluster is queried separately to test if the job is running.
    on either cluster.

    Raises RuntimeError if no login node answers, so a transient SSH failure
    can't masquerade as 'not running' and cause duplicate submissions.

    Args:
        job_name (str): Name of the job to check

    Returns:
        bool: True if the job is pending or running, False otherwise
    """
    hosts = ["alex.nhr.fau.de", "tinyx.nhr.fau.de"]
    sq = 'squeue -t PENDING,RUNNING --format "%200j" -h'
    ssh_opts = "-o ConnectTimeout=5 -o BatchMode=yes"

    answered = False
    for host in hosts:
        result = local(
            f"ssh {ssh_opts} {env.user}@{host} '{sq}'; echo __SQ_EXIT__$?",
            capture=True,
        )
        text = str(result) if result else ""
        marker_idx = text.rfind("__SQ_EXIT__")
        if marker_idx < 0:
            print(
                f"[warn] squeue probe on {host} produced no marker; skipping"
            )
            continue
        exit_code = text[marker_idx + len("__SQ_EXIT__") :].strip()
        if exit_code != "0":
            print(
                f"[warn] squeue probe on {host} exited {exit_code}; skipping"
            )
            continue
        answered = True
        body = text[:marker_idx]
        jobs = [line.strip() for line in body.splitlines() if line.strip()]
        if job_name in jobs:
            return True

    if not answered:
        raise RuntimeError(
            f"Could not query squeue on any login node ({hosts}); aborting to avoid duplicate submission of {job_name}"
        )
    return False


def check_dir_exists(dir_path):
    """Check if a directory exists on the remote cluster.

    Raises RuntimeError on ambiguous probe output for the same reason as
    check_file_exists — we never want 'unknown' coerced to 'absent'.

    Args:
        dir_path (str): Absolute path to the directory to check

    Returns:
        bool: True if directory exists, False otherwise
    """
    result = run(
        f"test -d {dir_path} && echo __DE__1 || echo __DE__0",
        quiet=True,
        warn_only=True,
    )
    text = str(result) if result is not None else ""
    if getattr(result, "failed", False):
        raise RuntimeError(
            f"check_dir_exists probe failed for {dir_path}: {text!r}"
        )
    if "__DE__1" in text:
        return True
    if "__DE__0" in text:
        return False
    raise RuntimeError(
        f"check_dir_exists got ambiguous output for {dir_path}: {text!r}"
    )


def check_file_exists(file_path):
    """Check if a file exists on the remote cluster.

    Raises RuntimeError if the probe cannot answer cleanly (SSH error, empty
    output) — treating "unknown" as "absent" caused silent fresh-starts that
    overwrote resumable checkpoints.

    Args:
        file_path (str): Path to the file to check

    Returns:
        bool: True if file exists, False otherwise
    """
    result = run(
        f"test -f {file_path} && echo __FE__1 || echo __FE__0",
        quiet=True,
        warn_only=True,
    )
    text = str(result) if result is not None else ""
    if getattr(result, "failed", False):
        raise RuntimeError(
            f"check_file_exists probe failed for {file_path}: {text!r}"
        )
    if "__FE__1" in text:
        return True
    if "__FE__0" in text:
        return False
    raise RuntimeError(
        f"check_file_exists got ambiguous output for {file_path}: {text!r}"
    )


def _find_latest_last_ckpt(ckpt_dir):
    """Return the newest-by-mtime ``last*.ckpt`` in ``ckpt_dir``, or None.

    Lightning's ``ModelCheckpoint(save_last=True)`` emits versioned
    ``last-v{N}.ckpt`` when ``last.ckpt`` already exists from a prior run,
    so ``last.ckpt`` ends up frozen at the first run's epoch. Picking the
    newest real save by mtime recovers the actual latest state.

    Raises RuntimeError if the probe can't answer cleanly — same reasoning
    as check_file_exists: never let "unknown" masquerade as "absent".

    Args:
        ckpt_dir (str): Absolute path to the checkpoints directory.

    Returns:
        str | None: Full path to newest last*.ckpt, or None if none exist.
    """
    probe = (
        f"find {ckpt_dir} -maxdepth 1 -type f -name 'last*.ckpt' "
        f"-printf '%T@\\t%p\\n' 2>/dev/null | sort -rn | head -1 | cut -f2-; "
        f"echo __FL__$?"
    )
    result = run(probe, quiet=True, warn_only=True)
    text = str(result) if result is not None else ""
    marker_idx = text.rfind("__FL__")
    if marker_idx < 0:
        raise RuntimeError(
            f"_find_latest_last_ckpt probe produced no marker "
            f"for {ckpt_dir}: {text!r}"
        )
    exit_code = text[marker_idx + len("__FL__") :].strip()
    body = text[:marker_idx].strip()
    if exit_code != "0":
        raise RuntimeError(
            f"_find_latest_last_ckpt probe failed (exit {exit_code}) "
            f"for {ckpt_dir}: {text!r}"
        )
    return body or None


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


def _remove_completion_markers(test_root):
    """Remove COMPLETE and LAST_RUN markers from all test set subdirectories.

    This allows a previously completed experiment to be re-evaluated. Existing
    results directories are preserved (new runs get a fresh timestamp).
    """
    cmd = f"""
test_root="{test_root}"
if [[ -d "$test_root" ]]; then
    find "$test_root" -mindepth 2 -maxdepth 2 -name COMPLETE -delete 2>/dev/null
    find "$test_root" -mindepth 2 -maxdepth 2 -name LAST_RUN -delete 2>/dev/null
fi
"""
    run(cmd, quiet=True)


# Per-cluster GPU assignment: dataset -> {cluster -> gpu_partition}
_GPU_MAP = {
    "datasets/cnceleb": {"alex": "a40", "tinygpu": "v100"},
    "datasets/voxceleb": {"alex": "a100", "tinygpu": "v100"},
    "multi_sv": {"alex": "a40", "tinygpu": "a100"},
    "multi_sv_cnc_train": {"alex": "a100", "tinygpu": "a100"},
}


def _get_job_routing(experiment, dataset_name, sparsity=None):
    """Return (cluster_name, gpu_type) for a job.

    Targeting ~80% alex / ~20% tinygpu when all experiment types are active:
    - ProxSGD -> tinygpu (baseline comparison, ~1/5 of experiment types)
    - Rescale Prox v2 (lambda/prev_lambda) -> tinygpu
    - All other Bregman/pruning -> alex
    - Baselines (sparsity=None) -> current cluster (run on whichever is active)
    """
    if sparsity is None:
        cluster = CLUSTER_NAME
    elif "proxsgd" in experiment:
        cluster = "tinygpu"
    elif "rescale_prox_v2" in experiment:
        cluster = "tinygpu"
    elif "subgrad_corr_v3" in experiment:
        cluster = "tinygpu"
    elif "fixed" in experiment:
        cluster = "tinygpu"
    else:
        cluster = "alex"

    gpu = _GPU_MAP[dataset_name][cluster]
    return cluster, gpu


def _submit_sv_job(
    # hparams
    experiment,
    sv_model,
    dataset_name,
    transfer_data_bool,
    max_epochs,
    apply_augmentation=False,
    batch_size_base=128,
    target_sparsity=None,
    log_every_n_steps=1,
    # progressive ramp (Bregman only)
    progressive=False,
    initial_target_sparsity=None,
    ramp_epochs=None,
    # other options
    gpu_device=GPU,
    bypass_exps=None,
    force_retest=False,
    extra_overrides=None,
    job_name_suffix="",
):
    """Helper function to configure and submit a single SV training job."""
    if not apply_augmentation:
        # augmentation doubles the effective batch via paired views; match it
        batch_size_base *= 2
    apply_vad = False
    schedule_type = "constant"  # Options: 'constant', 'linear'
    num_ckpt_avg = 0  # Number of checkpoints to average for final model (only applicable for certain experiments)
    ramp_up_epochs = 10
    virtual_spks = "False"
    logger = "many_loggers"  # options tensorboard , many_loggers

    settings = {
        "script_name": "src/train.py",
        "cluster": CLUSTER_NAME,
        "path_project": PATH_PROJECT,
        "env_name": CONDA_ENV,
        "data_path": DATA_DIR,
        "gpu": gpu_device,
        "num_nodes": 1,
        "walltime": "24:00:00",
        "num_gpus": 1,
        "cuda": "12.9.0",
    }
    ramp_up_experiments = [
        "sv_pruning_mag_struct",
        "sv_pruning_mag_unstruct",
    ]
    large_models = [
        "wespeaker_resnet293",
        "wespeaker_pretrained_resnet293",
        "wespeaker_resnet152",
        "wespeaker_pretrained_resnet152",
    ]

    bypass_exps = bypass_exps or []
    is_pretrained = "pretrained" in sv_model
    is_onetime = "onetime" in experiment

    # Specify sparsity if not baseline
    if "bregman" in experiment or "pruning" in experiment:
        assert (
            target_sparsity is not None
        ), "target_sparsity should be set for pruning and bregman experiments"
    else:
        assert (
            target_sparsity is None
        ), "target_sparsity should be None for baseline experiments"

    # skip when model type and experiment type disagree
    if is_pretrained != is_onetime:
        if is_pretrained:
            print(
                f"\nSkipping {experiment} with {sv_model}  (Should be trained from scratch only)!\n"
            )
        else:
            print(
                f"\nSkipping {experiment} with {sv_model}  (onetime pruning is for fine-tuning)!\n"
            )
        return

    # Handling epochs
    epochs_to_ramp = None
    if experiment in ramp_up_experiments:
        epochs_to_ramp = ramp_up_epochs
        max_epochs += epochs_to_ramp

    # Handling of special cases:
    batch_size = 16 if sv_model in large_models else batch_size_base

    # cannot use pruning with certain models
    if "pruning" in experiment:
        if sv_model in (
            "speechbrain_pretrained_ecapa_tdnn",
            "pretraiend_ecapa2",
        ):
            print(
                f"\nSkipping {experiment} with {sv_model} - pruning not supported for this model!!\n"
            )
            return

    # shared settings
    ramp_str = (
        f"-ramp{epochs_to_ramp}_{schedule_type}" if epochs_to_ramp else ""
    )
    sparsity_str = (
        f"-sr{int(target_sparsity * 100)}"
        if target_sparsity is not None
        else ""
    )
    progressive_str = (
        f"-ramp{int(initial_target_sparsity * 100)}to"
        f"{int(target_sparsity * 100)}_{ramp_epochs}ep"
        if progressive
        and initial_target_sparsity is not None
        and target_sparsity is not None
        and ramp_epochs is not None
        else ""
    )
    num_ckpt_avg_str = f"-avg{num_ckpt_avg}" if num_ckpt_avg > 1 else ""

    job_name = (
        f"{experiment}{ramp_str}-{sv_model}-{os.path.basename(dataset_name)}"
        f"-virt-{virtual_spks}-bs{batch_size}-vad{apply_vad}"
        f"{num_ckpt_avg_str}-ep{max_epochs}-aug{apply_augmentation}"
        f"{sparsity_str}{progressive_str}{job_name_suffix}"
    )

    settings = settings.copy()
    settings["job_name"] = job_name
    name = f"{os.path.basename(dataset_name)}/{job_name}"

    script_arguments = {
        "datamodule": dataset_name,
        "experiment": f"sv/{experiment}",
        "module": "sv",
        "trainer": "gpu",
        "name": name,
        "module/sv_model": sv_model,
        "logger": logger,
        "datamodule.loaders.train.batch_size": batch_size,
        "datamodule.loaders.valid.batch_size": batch_size,
        "datamodule.loaders.enrollment.batch_size": 8,
        "datamodule.loaders.test.batch_size": 128,
        "trainer.max_epochs": max_epochs,
        "paths.log_dir": RESULTS_DIR,
        "hydra.run.dir": f"{RESULTS_DIR}/train/runs/{name}",
        "trainer.num_sanity_val_steps": 0,
        "trainer.log_every_n_steps": log_every_n_steps,
    }

    # Disable checkpoint averaging if num_ckpt_avg is 1 or less
    if num_ckpt_avg > 1:
        script_arguments[
            "callbacks.checkpoint_averaging.num_checkpoints"
        ] = num_ckpt_avg
    else:
        script_arguments["callbacks.checkpoint_averaging"] = "null"

    # if dataset_name != "multi_sv" or dataset_name != "multi_sv_cnc_train":
    #     script_arguments["datamodule.dataset.vad.enabled"] = str(apply_vad)

    if epochs_to_ramp:
        script_arguments.update(
            {
                "callbacks.model_pruning.epochs_to_ramp": epochs_to_ramp,
                "callbacks.model_pruning.schedule_type": schedule_type,
            }
        )

    if not apply_augmentation:
        # if experiment not in no_aug_exps:
        script_arguments["module.data_augmentation"] = "null"
    else:
        script_arguments[
            "module.data_augmentation.augmentations.wav_augmenter.speed_perturb.virtual_speakers"
        ] = virtual_spks

    if target_sparsity is not None:
        if "bregman" in experiment:
            script_arguments["_bregman_target_sparsity"] = target_sparsity
        elif "pruning" in experiment:
            script_arguments[
                "callbacks.model_pruning.amount"
            ] = target_sparsity
            script_arguments[
                "callbacks.model_pruning.final_amount"
            ] = target_sparsity
            script_arguments[
                "callbacks.model_pruning.save_when_sparser_than"
            ] = target_sparsity

    if extra_overrides:
        script_arguments.update(extra_overrides)

    # Jobs submission logic
    if check_running_pending(job_name):
        print(f"Skipping {job_name} - already running or pending")
        return

    run_dir = script_arguments["hydra.run.dir"]
    # Check if test artifacts already exist to avoid re-running testing
    if not force_retest and check_test_artifacts_complete(run_dir):
        print(f"Skipping {job_name} - testing already complete")
        return

    if run_dir in bypass_exps:
        print(f"Skipping {job_name} - in bypass list for testing")
        return

    # If a checkpoint exists, resume from it. This can happen when a job was previously running and got interrupted.
    # Lightning's save_last sometimes writes last-v{N}.ckpt instead of
    # overwriting last.ckpt, so last.ckpt can be stale — pick newest by mtime.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest_last = _find_latest_last_ckpt(ckpt_dir)
    if latest_last:
        script_arguments["ckpt_path"] = latest_last
        latest_name = os.path.basename(latest_last)
        if latest_name != "last.ckpt":
            print(
                f"[warn] {job_name}: last.ckpt is stale — "
                f"resuming from {latest_name} instead"
            )
        print(f"Resuming {job_name} from {latest_name}")
        if any(lg in logger for lg in ["wandb"]):
            script_arguments["logger.wandb.id"] = name
            print(f"Resuming run in wandb dashboard with id: {name}")
    elif check_dir_exists(ckpt_dir):
        raise RuntimeError(
            f"Checkpoints dir exists but no last*.ckpt found for {job_name}.\n"
            f"  dir: {ckpt_dir}\n"
            f"Refusing to start from scratch and overwrite existing checkpoints."
        )

    if force_retest:
        script_arguments["+module.force_retest"] = True
        # Remove completion markers so the job starts fresh
        test_root = os.path.join(run_dir, "test_artifacts")
        _remove_completion_markers(test_root)
        print(f"Force retest: cleared completion markers for {job_name}")

    bash_script = create_sv_bash_script(
        settings, script_arguments, transfer_data=transfer_data_bool
    )
    run_bash_script(bash_script)
    print(f"Submitted job: {job_name} (transfer_data={transfer_data_bool})")
    time.sleep(0.1)


@task
def run_sv(transfer_data="false", force="false"):
    """Generate and submit all SV training jobs for multi_sv dataset.

    Bregman and iterative-pruning experiments sweep over all sparsity rates.
    One-time pruning and baseline experiments run at the config default.

    Args:
        transfer_data (str): 'true' to transfer data to local SSD, 'false' to use shared filesystem
        force (str): 'true' to force re-evaluation of already-completed experiments
    """
    transfer_data_bool = transfer_data.lower() in ("true", "1", "yes")
    force_retest = force.lower() in ("true", "1", "yes")

    batch_sizes = {
        "wespeaker_ecapa_tdnn": 128,
        "wespeaker_resnet34": 64,
        "wespeaker_resnet50": 32,
    }

    # Default hparams shared across experiments
    ecapa = "wespeaker_ecapa_tdnn"
    resnet34 = "wespeaker_resnet34"
    default_sv_models = [ecapa, resnet34]
    default_sparsity_rates = [0.90, 0.95, 0.99]
    dataset_names = ["datasets/cnceleb", "multi_sv"]

    base_max_epochs = {
        "datasets/cnceleb": 40,
        "datasets/voxceleb": 20,
        "multi_sv": 20,
        "multi_sv_cnc_train": 20,
    }

    ########################
    # Switch controls (master switch per group of experiments)
    ########################
    RUN_BASELINE_EXPS = False
    RUN_PRUNING_EXPS = False
    RUN_Bregman_EXPS = False
    RUN_AUX_BREGMAN_EXPS = False
    # Bregman cells: {ramp, fixed} x {trainable scales, uniform}.
    RUN_PROGRESSIVE_ONLY_EXPS = True  # ramp,  uniform
    RUN_TRAINABLE_FIXED_EXPS = True  # fixed, trainable
    RUN_PROGRESSIVE_TRAINABLE_EXPS = False  # ramp,  trainable
    RUN_VANILLA_FIXED_EXPS = False  # fixed, uniform

    ########################
    # Experiment registry: a list of entries (same config may recur with
    # different overrides, e.g. cases 2 and 3).
    #   required: experiment, sv_models, dataset_names, sparsity_rates
    #   optional: extra_overrides, suffix, per_model,
    #             initial_target_sparsity + ramp_epochs (=> progressive ramp)
    ########################
    EXPERIMENTS = []

    # Baselines: no sparsity target, swept over default models/datasets.
    if RUN_BASELINE_EXPS:
        for _exp in ("sv_wespeaker", "sv_vanilla"):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": default_sv_models,
                    "dataset_names": dataset_names,
                    "sparsity_rates": [None],
                }
            )

    # Magnitude pruning (sv_pruning_mag_struct also available).
    if RUN_PRUNING_EXPS:
        EXPERIMENTS.append(
            {
                "experiment": "sv_pruning_mag_unstruct",
                "sv_models": default_sv_models,
                "sparsity_rates": default_sparsity_rates,
                "dataset_names": dataset_names,
            }
        )

    # Bregman: both models use RegL1 on conv (set in the experiment configs).
    if RUN_Bregman_EXPS:
        for _exp in ("sv_bregman_linbreg", "sv_bregman_adabreg"):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa, resnet34],
                    "sparsity_rates": [0.95, 0.99],
                    "dataset_names": ["multi_sv"],
                    "suffix": "-regl1_conv",
                }
            )

    # Auxiliary fixed-lambda Bregman (static lambda, no scheduler).
    if RUN_AUX_BREGMAN_EXPS:
        for _exp in ("sv_bregman_adabreg_fixed", "sv_bregman_linbreg_fixed"):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa, resnet34],
                    "sparsity_rates": [0.90],
                    "dataset_names": ["datasets/cnceleb"],
                }
            )

    # CASE 1 — progressive ramp, uniform allocation.
    if RUN_PROGRESSIVE_ONLY_EXPS:
        for _exp in (
            "sv_bregman_adabreg_progressive",
            "sv_bregman_linbreg_progressive",
        ):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa],
                    "sparsity_rates": [0.95, 0.99],
                    "dataset_names": ["multi_sv"],
                    "initial_target_sparsity": 0.0,
                    "ramp_epochs": 10,
                    "ramp_granularities": ["epoch", "step"],
                }
            )

    # CASE 2 — trainable scales at a fixed target.
    if RUN_TRAINABLE_FIXED_EXPS:
        for _exp in (
            "sv_bregman_adabreg_trainable_scales",
            "sv_bregman_linbreg_trainable_scales",
        ):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa, resnet34],
                    "sparsity_rates": [0.99],
                    "dataset_names": ["multi_sv"],
                    "suffix": "-fixed_target",
                }
            )

    # CASE 3 — trainable scales WITH a progressive ramp.
    if RUN_PROGRESSIVE_TRAINABLE_EXPS:
        for _exp in (
            "sv_bregman_adabreg_trainable_scales",
            "sv_bregman_linbreg_trainable_scales",
        ):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa, resnet34],
                    "sparsity_rates": [0.99],
                    "dataset_names": ["multi_sv"],
                    "initial_target_sparsity": 0.0,
                    "ramp_epochs": 10,
                    "ramp_granularities": ["epoch", "step"],
                }
            )

    # CASE 4 — vanilla Bregman at a fixed target, uniform allocation. Dense start
    # to match the other cells (base defaults to 0.99 sparse).
    if RUN_VANILLA_FIXED_EXPS:
        for _exp in ("sv_bregman_adabreg", "sv_bregman_linbreg"):
            EXPERIMENTS.append(
                {
                    "experiment": _exp,
                    "sv_models": [ecapa],
                    "sparsity_rates": [0.99],
                    "dataset_names": ["multi_sv"],
                    "extra_overrides": {"_bregman_initial_sparsity": 0.0},
                    "suffix": "-fixed_target",
                }
            )

    # --- Volume estimation across clusters ---
    job_counts = {
        "alex": {"a40": 0, "a100": 0},
        "tinygpu": {"v100": 0, "a100": 0},
    }
    for cfg in EXPERIMENTS:
        experiment = cfg["experiment"]
        n_variants = len(cfg.get("ramp_granularities", [None]))
        for dataset_name in cfg["dataset_names"]:
            for sparsity in cfg["sparsity_rates"]:
                cluster, gpu = _get_job_routing(
                    experiment, dataset_name, sparsity
                )
                job_counts[cluster][gpu] += len(cfg["sv_models"]) * n_variants

    total_jobs = sum(sum(gpus.values()) for gpus in job_counts.values())
    print(f"\n{'='*50}")
    print(f"Job distribution ({total_jobs} total):")
    for cname, gpus in job_counts.items():
        total = sum(gpus.values())
        if total == 0:
            continue
        gpu_breakdown = ", ".join(
            f"{g}: {n}" for g, n in gpus.items() if n > 0
        )
        pct = 100 * total / total_jobs if total_jobs else 0
        marker = " <-- current" if cname == CLUSTER_NAME else ""
        print(
            f"  {cname}: {total} jobs ({pct:.0f}%) [{gpu_breakdown}]{marker}"
        )
    print(f"{'='*50}\n")

    # --- Submit all experiments (only for current cluster) ---
    for cfg in EXPERIMENTS:
        experiment = cfg["experiment"]
        base_overrides = dict(cfg.get("extra_overrides") or {})
        base_suffix = cfg.get("suffix", "")
        per_model_cfg = cfg.get("per_model", {})

        # Pin the schedule explicitly: progressive entries ramp; other Bregman
        # entries pin a fixed target (never relying on the config default).
        prog = {}
        if "ramp_epochs" in cfg:
            variants = cfg.get("ramp_granularities", ["epoch"])
            sched_overrides = {
                "_bregman_target_initial": cfg["initial_target_sparsity"],
                "_bregman_ramp_epochs": cfg["ramp_epochs"],
            }
            prog = {
                "progressive": True,
                "initial_target_sparsity": cfg["initial_target_sparsity"],
                "ramp_epochs": cfg["ramp_epochs"],
            }
        elif "bregman" in experiment:
            variants = [None]
            sched_overrides = {
                "_bregman_target_initial": "null",
                "_bregman_ramp_epochs": "null",
            }
        else:
            variants = [None]
            sched_overrides = {}

        for sv_model in cfg["sv_models"]:
            pm = per_model_cfg.get(sv_model, {})
            model_overrides = {
                **base_overrides,
                **sched_overrides,
                **(pm.get("extra_overrides") or {}),
            }
            model_suffix = pm.get("suffix", base_suffix)

            for variant in variants:
                if variant is None:
                    extra_overrides = model_overrides
                    suffix = model_suffix
                else:
                    extra_overrides = {
                        **model_overrides,
                        "_bregman_ramp_granularity": variant,
                    }
                    suffix = f"{model_suffix}-{variant}wise"
                variant_prog = prog

                for dataset_name in cfg["dataset_names"]:
                    for sparsity in cfg["sparsity_rates"]:
                        cluster, gpu = _get_job_routing(
                            experiment, dataset_name, sparsity
                        )
                        if cluster != CLUSTER_NAME:
                            print(
                                f"Skipping {experiment} with {sv_model} on {dataset_name} - routed to {cluster} cluster"
                            )
                            continue
                        _submit_sv_job(
                            experiment=experiment,
                            sv_model=sv_model,
                            dataset_name=dataset_name,
                            batch_size_base=batch_sizes[sv_model],
                            transfer_data_bool=transfer_data_bool,
                            max_epochs=base_max_epochs[dataset_name],
                            target_sparsity=sparsity,
                            gpu_device=gpu,
                            force_retest=force_retest,
                            extra_overrides=extra_overrides or None,
                            job_name_suffix=suffix,
                            **variant_prog,
                        )


def create_eval_bash_script(settings, script_arguments):
    """Creates a SLURM bash script for SV evaluation jobs.

    Unlike training scripts, no data transfer is performed — evaluation reads
    directly from the shared filesystem via the pre-built exp_dir.

    Args:
        settings (dict): Dictionary containing job configuration settings.
        script_arguments (dict): Dictionary of arguments to pass to eval.py.

    Returns:
        str: A bash script as a string, ready to be submitted to SLURM.
    """
    script_arguments_str = " ".join(
        [f"{k}={v}" for k, v in script_arguments.items()]
    )
    return f"""#!/bin/bash -l
#SBATCH --job-name={settings['job_name']}
#SBATCH --clusters={settings['cluster']}
#SBATCH --partition={settings['gpu']}
#SBATCH --nodes={settings['num_nodes']}
#SBATCH --gres=gpu:{settings['gpu']}:{settings['num_gpus']}
#SBATCH --time={settings['walltime']}
#SBATCH --export=NONE

module load cuda/{settings['cuda']}
source ~/miniconda3/bin/activate {settings['env_name']}
cd {settings['path_project']}

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

echo 'Starting evaluation'
python {settings['script_name']} {script_arguments_str}
"""


@task
def eval_sv(force="true"):
    """Submit evaluation jobs for trained SV models.

    Args:
        force (str): 'true' to force re-evaluation of already-completed experiments
    """
    force_retest = force.lower() in ("true", "1", "yes")

    EVAL_OUTPUT_DIR = os.path.join(
        HPC_HOME_DIR, "adversarial-robustness-for-sr/logs/eval/runs"
    )
    TRAINED_MODELS_DIR = os.path.join(RESULTS_DIR, "train/runs")

    exp_paths = [
        ############
        # CNCELEB
        ############
        # baselines
        # "sv_vanilla-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs40-data_augFalse",
        # "sv_wespeaker-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs40-data_augFalse",
        # Pruning at 90% sparsity
        "sv_pruning_mag_unstruct-ramp10_constant-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs50-data_augFalse",
        "sv_pruning_mag_struct-ramp10_constant-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs50-data_augFalse",
        # Pruning at 95% sparsity
        "sv_pruning_mag_unstruct-ramp10_constant-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs50-data_augFalse-sparsity95",
        "sv_pruning_mag_struct-ramp10_constant-wespeaker_ecapa_tdnn-cnceleb-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs50-data_augFalse-sparsity95",
        ###############
        # VOXCELEB
        ###############
        # baselines
        "sv_wespeaker-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs10-data_augFalse",
        "sv_vanilla-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs20-data_augFalse",
        # Pruning at 90% sparsity
        "sv_pruning_mag_unstruct-ramp10_constant-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs30-data_augFalse",
        "sv_pruning_mag_struct-ramp10_constant-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs30-data_augFalse",
        # Pruning at 95% sparsity
        "sv_pruning_mag_unstruct-ramp10_constant-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs30-data_augFalse-sparsity95",
        "sv_pruning_mag_struct-ramp10_constant-wespeaker_ecapa_tdnn-multi_sv-virtual_spks-False-bs256-vadFalse-ckpt_avg3-max_epochs30-data_augFalse-sparsity95",
    ]

    base_settings = {
        "script_name": "src/eval.py",
        "cluster": CLUSTER_NAME,
        "path_project": PATH_PROJECT,
        "env_name": CONDA_ENV,
        "gpu": "v100" if CLUSTER_NAME == "tinygpu" else "a40",
        "num_nodes": 1,
        "walltime": "24:00:00",
        "num_gpus": 1,
        "cuda": "12.9.0",
    }

    for path in exp_paths:

        job_name = f"eval-{path}"

        # Extract dataset from exp name (assuming it contains either "cnceleb", "voxceleb", or neither for multi_sv)
        dataset = (
            "cnceleb"
            if "cnceleb" in path
            else ("voxceleb" if "voxceleb" in path else "multi_sv")
        )
        exp_dir = os.path.join(TRAINED_MODELS_DIR, dataset, path)
        print(f"Preparing evaluation for {exp_dir}")

        # Extract valid metric from filename and sort by best metric, then epoch as tiebreaker.
        # Supports both old and new naming schemes:
        #   old: epoch{epoch}-loss_valid{loss}-metric_valid{metric}.ckpt
        #   new: epoch{epoch}-loss_valid{loss}-metric_valid{metric}-sr{sr}.ckpt
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        result = run(
            f"ls -1 {ckpt_dir}/epoch*.ckpt 2>/dev/null || true", quiet=True
        )
        ckpt_names = [
            line.strip() for line in result.splitlines() if line.strip()
        ]
        if not ckpt_names:
            raise ValueError(f"No checkpoints found for {path}")

        def _ckpt_sort_key(p):
            name = os.path.basename(p).removesuffix(".ckpt")
            parts = name.split("-")
            epoch = int(parts[0].removeprefix("epoch"))
            metric = float(parts[2].removeprefix("metric_valid"))
            return (metric, epoch)

        ckpt_names.sort(key=_ckpt_sort_key, reverse=True)
        best_ckpt = ckpt_names[0]

        if check_running_pending(job_name):
            print(f"Skipping {job_name} - already running or pending")
            continue

        if not force_retest and check_test_artifacts_complete(exp_dir):
            print(f"Skipping {path} - testing already complete")
            continue

        if force_retest:
            test_root = os.path.join(exp_dir, "test_artifacts")
            _remove_completion_markers(test_root)
            print(f"Force retest: cleared completion markers for {job_name}")

        settings = base_settings.copy()
        settings["job_name"] = job_name

        script_arguments = {
            "exp_dir": exp_dir,
            "datamodule.loaders.train.batch_size": 8,
            "datamodule.loaders.enrollment.batch_size": 8,
            "datamodule.loaders.test.batch_size": 128,
            "datamodule.dataset.enrollment_mode": "both",
            "ckpt_path": best_ckpt,
            "paths.data_dir": f'"{DATA_DIR}"',
            # This will automatically resume the test if it was previously interrupted
            "paths.log_dir": os.path.join(EVAL_OUTPUT_DIR, dataset),
            "hydra.run.dir": os.path.join(EVAL_OUTPUT_DIR, dataset, path),
        }

        if force_retest:
            script_arguments["+module.force_retest"] = True

        bash_script = create_eval_bash_script(settings, script_arguments)
        run_bash_script(bash_script)
        print(f"Submitted eval job: {job_name}")
        time.sleep(0.1)
