import os
import datetime
import time

from fabric.api import run, env, cd, task
from fabric.contrib.project import rsync_project

# Cluster configuration
env.user = 'dsnf101h' # 'iwal021h'

CLUSTER_NAME = 'tinygpu'    # Options: 'tinygpu', 'alex'
env.hosts = ['alex.nhr.fau.de'] if CLUSTER_NAME == 'alex' else ['tinyx.nhr.fau.de']
GPU = 'a100'  # Options: 'rtx2080ti', 'rtx3080', 'a100'

# Path configuration
PATH_PROJECT = '~/adversarial-robustness-for-sr'  # project folder on the hpc cluster
CONDA_ENV = 'comfort'

WOODY_DIR = f'/home/woody/{env.user[: 4]}/{env.user}'
RESULTS_DIR = os.path.join(WOODY_DIR, 'results')
DATA_DIR = os.path.join(WOODY_DIR, 'datasets')

# sync_results paths
# SYNC_DIR_REMOTE = os.path.join(RESULTS_DIR, 'train/runs/*/vpc_amm_cyclic-*-max_dur10-bs32')  # remote results dir
SYNC_DIR_REMOTE = os.path.join(RESULTS_DIR, 'train/runs/available_models/*available_models**')  # remote results dir
SYNC_DIR_LOCAL = '/dataHDD/ahmad/hpc_results/libri_augmented3/'  # local results dir


def timestamp():
    """Return a formatted timestamp string for use in filenames or logs."""
    return str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')


@task
def sync_results():
    """
    Rsync the entire RESULTS_DIR from the HPC cluster to the local machine.
    """
    rsync_project(
        remote_dir=SYNC_DIR_REMOTE,
        local_dir=SYNC_DIR_LOCAL,
        exclude=['.DS_Store', '*.pyc', '__pycache__', '.git*', '*.ckpt', '*.pth', '.vscode'],
        delete=False,
        upload=False,
    )


def create_bash_script(settings, script_arguments):
    """
    Creates a bash script with parallelized data transfer for enhanced performance.
    
    Args:
        settings (dict): Dictionary containing job configuration settings.
        script_arguments (dict): Dictionary of arguments to pass to the training script.
    
    Returns:
        str: A bash script as a string, ready to be submitted to SLURM.
    """
    is_available_models = "available_models" in settings['datamodule_dir']
    is_libri_included = "LibriSpeech" in settings['datamodule_dir']
    script_arguments_str = ' '.join([f'{k}={v}' for k, v in script_arguments.items()])
    
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

VPC_PATH="{settings['data_path']}/{settings['vpc_dirname']}"
DEST_DIR="$TMPDIR/{settings['vpc_dirname']}"

# Function to transfer and extract model data
transfer_model() {{
    local model=$1
        
    local tar_file="$VPC_PATH/$model.tar.gz"
    [[ ! -f "$tar_file" ]] && echo "Error: $tar_file not found" && return 1
    
    echo "Transferring $model"
    rsync -ah "$tar_file" "$DEST_DIR/"
    tar -xzf "$DEST_DIR/$(basename "$tar_file")" -C "$DEST_DIR/"
    
    # Transfer metadata if needed
    if [[ "$model" != "librispeech" && -d "$VPC_PATH/$model/data/metadata" && ! -d "$DEST_DIR/$model/data/metadata" ]]; then
        rsync -ah "$VPC_PATH/$model/data/metadata" "$DEST_DIR/$model/data/"
    fi
    
    echo "Completed transfer of $model"
    return 0
}}

# Main data transfer
if [[ ! -d "$DEST_DIR" ]]; then
    mkdir -p "$DEST_DIR"
    
    if [[ "{is_available_models}" == "True" ]]; then
        echo "Transferring models in parallel"
        # Process B* and T* models
        for model_tar in "$VPC_PATH"/{{B*,T*}}.tar.gz; do
            [[ -f "$model_tar" ]] && transfer_model "$(basename "$model_tar" .tar.gz)" &
        done
        
        # Transfer librispeech for multiple models
        [[ "{is_libri_included}" == "True" ]] && transfer_model "librispeech" &

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
    echo "All data transfers complete"
else
    echo "Data already exists, waiting briefly to avoid conflicts"
    sleep 9m
fi

# Start training
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

echo 'Starting training'
python {settings['script_name']} {script_arguments_str} paths.data_dir=$TMPDIR
"""
    return job_string


def create_bash_script_sv(settings, script_arguments):
    """
    Creates a bash script with parallelized data transfer for enhanced performance.
    
    Args:
        settings (dict): Dictionary containing job configuration settings.
        script_arguments (dict): Dictionary of arguments to pass to the training script.
    
    Returns:
        str: A bash script as a string, ready to be submitted to SLURM.
    """
    script_arguments_str = ' '.join([f'{k}={v}' for k, v in script_arguments.items()])
    
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

echo "No Data transfer is conducted for VoxCeleb dataset"

# Start the training process
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

echo 'Starting training'
python {settings['script_name']} {script_arguments_str} paths.data_dir=/home/woody/dsnf/dsnf101h/datasets
"""
    return job_string


def run_bash_script(pbs):
    with cd(PATH_PROJECT):
        run(f'PATH=$PATH:/apps/slurm/current/bin && cat << "EOF" |  sbatch\n{pbs}\nEOF', shell=True)


@task
def scancel():
    """Stop all jobs associated to your username."""
    run("squeue | grep $USER | tr -s ' ' | cut -d ' ' -f 2 | xargs scancel")


@task
def squeue():
    """Get the queue status of the tinygpu cluster."""
    run('squeue -l') # -l: long report


@task
def clean_logs():
    """Delete log files."""
    with cd(PATH_PROJECT):
        run('rm *.o*')
        run('rm *.e*')


@task
def cmd(cmd):
    """Wraps the command over 'run'."""
    run(cmd)


@task
def lsp():
    """Returns the names of pending jobs on the queue."""
    print(run('squeue -t PENDING --format "%.j" -h').split())

@task
def lsr():
    """Returns the names of Running jobs ."""
    print(run('squeue -t RUNNING --format "%.j" -h').split())


@task
def lsa():
    """Returns the names of all jobs."""
    print(run('squeue --format "%.j" -h').split())


def check_running_pending(job_name):
    """Check if a job is pending or running.
    
    Returns:
        int: 1 if the job is pending or running, 0 otherwise
    """
    files = run('squeue -t PENDING,RUNNING --format "%.j" -h').split()
    return 1 if job_name in files else 0


def check_file_exists(file_path):
    result = run(f'test -f {file_path} && echo 1 || echo 0')
    if result.stdout.strip() == '1':
        return 1
    else:
        return 0


@task
def run_vpc():
    """The main function that generates all VPC jobs """

    BATCH_SIZE = 32
    NUM_AUG = 1
    max_epochs = 25
    max_duration = 10
    dataset_dirname = 'vpc2025_official'

    settings = {
        'script_name': 'src/train.py',
        'cluster': CLUSTER_NAME,
        'path_project': PATH_PROJECT,
        'env_name': CONDA_ENV,
        'data_path': DATA_DIR,
        'vpc_dirname': dataset_dirname,
        'gpu': GPU,
        'num_nodes': 1,  # Multi-node are not possible on TinyGPU.
        'walltime': '24:00:00',
        'num_gpus': 1,
        'cuda': '11.1.0',  # '10.0'
    }

    # datasets = [
    #     "{LibriSpeech: ${datamodule.available_models.LibriSpeech}, B3: ${datamodule.available_models.B3}}",
    #     "{LibriSpeech: ${datamodule.available_models.LibriSpeech}}",
    #     "${datamodule.available_models}"
    #     ]
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
            
            # Parse dataset string like "{LibriSpeech: ...}, {B4: ...}" or a single dataset
            if dataset.startswith("{") and ":" in dataset:
                if ',' in dataset:
                    # Multiple datasets
                    dataset_name = '_'.join(item.split(':')[0].strip('{').strip() for item in dataset.split(','))
                else:
                    # Single dataset
                    dataset_name = dataset.split(':')[0].strip('{').strip()
            else:
                dataset_name = "available_models"

            job_name = experiment + '-' + dataset_name + '-' + 'max_dur' + str(max_duration) + '-' + 'bs' + str(BATCH_SIZE)

            # Defined this way to avoid re-training on different runs
            settings['job_name'] = job_name
            settings['datamodule_dir'] = dataset_dirname + os.sep + dataset_name
            name = dataset_name + os.sep + job_name

            script_arguments = {
                'datamodule': 'datasets/vpc',
                'experiment': f'vpc/{experiment}',
                'module': 'vpc',
                'trainer': 'gpu',
                'name': name,
                # 'logger': 'many_loggers.yaml',
                # 'logger.wandb.id': name,
                # 'logger.neptune.with_id': name,
                'datamodule.models': f"'{dataset}'",
                'datamodule.loaders.train.batch_size': BATCH_SIZE if 'aug' not in experiment else BATCH_SIZE // NUM_AUG,
                'datamodule.loaders.valid.batch_size': BATCH_SIZE if 'aug' not in experiment else BATCH_SIZE // NUM_AUG,
                'datamodule.dataset.max_duration': max_duration,
                'trainer.max_epochs': max_epochs,
                'paths.log_dir': f'{RESULTS_DIR}',
                'hydra.run.dir': f'{RESULTS_DIR}/train/runs/{name}',
                "trainer.num_sanity_val_steps": 0,
            }

            # Check for pending jobs: continue_flag = 1 if the job is pending
            continue_flag = check_running_pending(job_name)

            # 1- if pending: do nothing
            if continue_flag:
                continue

            # 2- if not: do the following:
            # get last_ckpt path
            last_ckpt = os.path.join(script_arguments['hydra.run.dir'], f'checkpoints/last.ckpt')

            last_ckpt_exists = check_file_exists(f'{last_ckpt}')
            if last_ckpt_exists:
                script_arguments['ckpt_path'] = last_ckpt

            bash_script = create_bash_script(settings, script_arguments)
            run_bash_script(bash_script)
            time.sleep(1.0)

@task
def run_sv():
    """The main function that generates all SV jobs """

    BATCH_SIZE = 32
    NUM_AUG = 1
    max_epochs = 10
    max_duration = 3.0
    dataset_dirname = 'voxceleb'

    settings = {
        'script_name': 'src/train.py',
        'cluster': CLUSTER_NAME,
        'path_project': PATH_PROJECT,
        'env_name': CONDA_ENV,
        'data_path': DATA_DIR,
        'vpc_dirname': dataset_dirname,
        'gpu': GPU,
        'num_nodes': 1,  # Multi-node are not possible on TinyGPU.
        'walltime': '24:00:00',
        'num_gpus': 1,
        'cuda': '11.1.0',  # '10.0'
    }

    experiments = ["sv_aug_prune", "sv_vanilla"]

    for experiment in experiments:

        dataset_name = "voxceleb"
        # batch_size = BATCH_SIZE if 'aug' not in experiment else BATCH_SIZE // NUM_AUG
        batch_size = BATCH_SIZE
        job_name = experiment + '-' + dataset_name + '-' + 'max_dur' + str(max_duration) + '-' + 'bs' + str(BATCH_SIZE)

        # Defined this way to avoid re-training on different runs
        settings['job_name'] = job_name
        settings['datamodule_dir'] = dataset_dirname + os.sep + dataset_name
        name = dataset_name + os.sep + job_name

        script_arguments = {
            'datamodule': 'datasets/voxceleb',
            'experiment': f'sv/{experiment}',
            'module': 'sv',
            'trainer': 'gpu',
            'name': name,
            # 'logger': 'many_loggers.yaml',
            # 'logger.wandb.id': name,
            # 'logger.neptune.with_id': name,
            'datamodule.loaders.train.batch_size': batch_size,
            'datamodule.loaders.valid.batch_size': batch_size,
            'datamodule.dataset.max_duration': max_duration,
            'trainer.max_epochs': max_epochs,
            'paths.log_dir': f'{RESULTS_DIR}',
            'hydra.run.dir': f'{RESULTS_DIR}/train/runs/{name}',
            "trainer.num_sanity_val_steps": 0,
        }

        # Check for pending jobs: continue_flag = 1 if the job is pending
        continue_flag = check_running_pending(job_name)

        # 1- if pending: do nothing
        if continue_flag:
            continue

        # 2- if not: do the following:
        # get last_ckpt path
        last_ckpt = os.path.join(script_arguments['hydra.run.dir'], f'checkpoints/last.ckpt')

        last_ckpt_exists = check_file_exists(f'{last_ckpt}')
        if last_ckpt_exists:
            script_arguments['ckpt_path'] = last_ckpt

        bash_script = create_bash_script_sv(settings, script_arguments)
        run_bash_script(bash_script)
        time.sleep(0.1)


if __name__ == '__main__':
    run_vpc()