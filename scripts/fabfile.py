import os
import datetime
import time
import uuid

from fabric.api import run, env, get, cd, task, hide
from fabric.contrib.project import rsync_project


env.user = 'iwal021h'
CLUSTER_NAME = 'alex'    #'tinygpu' 'alex'
env.hosts = ['alex.nhr.fau.de'] if CLUSTER_NAME == 'alex' else ['tinyx.nhr.fau.de']
GPU = 'a100'  #'rtx2080ti' 'rtx3080'

PATH_PROJECT = '~/adversarial-robustness-for-sr'  # project folder on the hpc cluster
CONDA_PATH = 'comfort' if CLUSTER_NAME == 'alex' else 'comfort_hpc'

WOODY_DIR = f'/home/woody/iwal/{env.user}'
RESULTS_DIR = WOODY_DIR + os.sep + 'results'
DATA_DIR = WOODY_DIR + os.sep + 'datasets'


def timestamp():
    return str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')

@task
def sync_results(expdir_name):
    """Rsync the project directory to the hpc."""
    rsync_project(
        remote_dir=PATH_PROJECT + os.sep + expdir_name + os.sep,
        local_dir=PATH_PROJECT + os.sep + expdir_name,
        exclude=['.DS_Store', '*.pyc', '__pycache__','.git*', '.vscode',],
        delete=False,
        upload=False,
    )

def create_bash_script(settings, script_arguments):
    """Creates a pbs file. Uses the scratch SSD for buffering training data."""

    script_arguments_str = ' '.join(['%s=%s' % (key, val) for key, val in script_arguments.items()])
    job_string = \
"""#!/bin/bash -l
#
# job name
#SBATCH --job-name={job_name}
#SBATCH --clusters={cluster}
#SBATCH --partition={gpu}
#SBATCH --nodes={num_nodes}
#SBATCH --gres=gpu:{gpu}:{num_gpus}
#SBATCH --time={walltime}
#SBATCH --export=NONE

# load cuda module first
module load cuda/{cuda}

# activate virtual environment
source ~/miniconda3/bin/activate {env_name}

# jobs always start in $HOME -
cd {path_project}

# transfer tarballs to the local SSD in $TMPDIR and untar it
# In case the data is transferred, wait for 15 minutes to avoid double unpacking of data
if ! [ -d "$TMPDIR/{datamodule_dir}" ]; then
mkdir $TMPDIR/{vpc_dirname}
time rsync -ahv {data_path}/{datamodule_dir}.tar.gz $TMPDIR/{vpc_dirname}/.
tar -xzf $TMPDIR/{datamodule_dir}.tar.gz -C $TMPDIR/{vpc_dirname}/.
rsync -ahv {data_path}/{datamodule_dir}/data/metadata $TMPDIR/{datamodule_dir}/data/.

echo "Data transferred to SSD"
elif [ -d "$TMPDIR/{datamodule_dir}" ]; then
sleep 0.15h
fi

# run script
echo 'Starting training'
python {script_name} {script_arguments} paths.data_dir=$TMPDIR

# nvidia-smi
""".format(**settings, script_arguments=script_arguments_str)

    return job_string

def run_bash_script(pbs):
    with cd(PATH_PROJECT):
        run('PATH=$PATH:/apps/slurm/current/bin && cat << "EOF" |  sbatch\n%s\nEOF' % pbs, shell=True)

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

def check_pending(job_name):
    files = run('squeue -t PENDING --format "%.j" -h').split()
    return 1 if job_name in files else 0

def check_file_exists(file_path):
    result = run(f'test -f {file_path} && echo 1 || echo 0')
    if result.stdout.strip() == '1':
        return 1
    else:
        return 0

@task
def run_exp():
    """The main function that generates all jobs """

    batch_size = 64
    max_epochs = 30
    datset_dir = 'vpc2025_official'

    settings = {
        'script_name': 'src/train.py',
        'cluster': CLUSTER_NAME,
        'path_project': PATH_PROJECT,
        'env_name': CONDA_PATH,
        'data_path': DATA_DIR,
        'vpc_dirname': datset_dir,
        'gpu': GPU,
        'num_nodes': 1,  # Multi-node are not possible on TinyGPU.
        'walltime': '24:00:00',
        'num_gpus': 1,
        'cuda': '11.1.0',  # '10.0'
    }

    datasets = ["{B3: ${datamodule.available_models.B3}}",
               "{B4: ${datamodule.available_models.B4}}",
               "{B5: ${datamodule.available_models.B5}}",
               "{T8-5: ${datamodule.available_models.T8-5}}",
               "{T12-5: ${datamodule.available_models.T12-5}}",
               "{T25-1: ${datamodule.available_models.T25-1}}",
               "${datamodule.available_models}"]
    classifiers = ["robust", "normalized"]
    loss_functions = ["fusion", "enhanced", "aam"]

    for dataset in datasets:
        for classifier in classifiers:
            for loss_function in loss_functions:
                dataset_name = dataset.split('.')[-1][:-2]
                classifier_name = classifier.split('=')[-1]
                loss_func_name = loss_function.split('=')[-1]
                job_name = 'classifier-' + classifier_name + '_' + 'loss-' + loss_func_name + '_' + dataset_name

                # Defined this way to avoid re-training on different runs
                settings['job_name'] = job_name
                settings['datamodule_dir'] = datset_dir + os.sep + dataset_name
                name = dataset_name + os.sep + job_name + '_' + str(batch_size)

                script_arguments = {
                    'datamodule': 'datasets/vpc',
                    'module': 'vpc',
                    'trainer': 'gpu',
                    'name': name,
                    'logger': 'neptune',
                    'logger.neptune.with_id': name,
                    'datamodule.models': f"'{dataset}'",
                    'datamodule.loaders.train.batch_size': batch_size,
                    'datamodule.loaders.valid.batch_size': batch_size,
                    'datamodule.dataset.max_duration': 12,
                    'module.model.classifiers.selected_classifier': f"{classifier}",
                    'module.criterion.selected_criterion': f"{loss_function}",
                    'trainer.max_epochs': max_epochs,
                    'paths.log_dir': f'{RESULTS_DIR}',
                    'hydra.run.dir': f'{RESULTS_DIR}/train/runs/{name}',
                    "+trainer.num_sanity_val_steps": 0

                }

                # Check for pending jobs: continue_flag = 1 if the job is pending
                continue_flag = check_pending(job_name)

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
                time.sleep(0.1)
                break
            break
        break

if __name__ == '__main__':
    run_exp()