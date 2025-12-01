# Robust Speaker Recognition Against Adversarial Attacks and Spoofing

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/gorodnitskiy/yet-another-lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2507.12081-orange)](https://arxiv.org/pdf/2507.12081)<br>

## Description

This repository bundles our research effort on **robust and efficient speaker recognition** across two related tasks:
- **Speaker verification** – training and evaluating modern architectures (ECAPA-TDNN, ResNets, custom WeSpeaker models) on public corpora such as VoxCeleb, LibriSpeech, and CN-Celeb.
- **Speaker De-anonymization** – against the [VoicePrivacy 2025](https://voiceprivacychallenge.org/) challenge data.

The framework builds on [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Hydra](https://github.com/facebookresearch/hydra) via [this template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template), which lets us compose experiments with declarative configs. Datasets live under `configs/datamodule/datasets` and include ready-made recipes for `voxceleb`, `cnceleb`, `librispeech`, and `vpc` (VoicePrivacy).
 
In addition, we also ship two compression techniques:
1. **Bregman Learning Framework** – adaptive regularization that induces sparsity during training (based on [Bungert et al. 2022](https://www.jmlr.org/papers/volume23/21-0545/21-0545.pdf) and the [TimRoith/BregmanLearning](https://github.com/TimRoith/BregmanLearning/tree/main/notebooks) reference implementation).
2. **Magnitude-Based Pruning** – structured or unstructured pruning with schedulers, checkpoint-safe masks, and deployment tooling.

📖 The compression was designed to experiment with the speaker recogntion models. However, they are implemented as Lightning callbacks, rendering their use flexible to other tasks. Learn how to enable these methods in **[docs/pruning.md](docs/pruning.md)**.


## Quick start

```shell
# clone template
git clone https://github.com/ahmad-aloradi/adversarial-robustness-for-sr.git
cd adversarial-robustness-for-sr

# install requirements
pip install -r requirements.txt
```

### Example: Override CLI arguments (Hydra style)

Hydra lets you override any config directly from the command line. The command below trains an ECAPA-TDNN model on CN-Celeb, switches to the structured pruning recipe, shrinks batch sizes, caps utterance duration, and limits the run length:

```bash
python src/train.py \
    datamodule=datasets/cnceleb \
    module/sv_model=wespeaker_ecapa_tdnn \
    experiment=sv/sv_pruning_mag_struct \
    datamodule.loaders.train.batch_size=8 \
    datamodule.loaders.valid.batch_size=8 \
    datamodule.dataset.max_duration=3.0 \
    trainer.max_epochs=10 \
    trainer.num_sanity_val_steps=1
```

Add further overrides (e.g., `logger=wandb`) as needed; Hydra composes them with the defaults defined under `configs/`.

## Main Packages

[PyTorch Lightning](https://github.com/Lightning-AI/lightning) - a lightweight deep learning framework / PyTorch
wrapper for professional AI researchers and machine learning engineers who need maximal flexibility without
sacrificing performance at scale.

[Hydra](https://github.com/facebookresearch/hydra) - a framework that simplifies configuring complex applications.
The key feature is the ability to dynamically create a hierarchical configuration by composition and override it
through config files and the command line.

## Project structure

- `src/`
- `data/`
- `logs/`
- `tests/`
- some additional directories, like: `notebooks/`, `docs/`, etc.

In this particular case, the directory structure looks like:

```
├── configs                     <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── debug                   <- Debugging configs
│   ├── experiment              <- Experiment configs
│   ├── extras                  <- Extra utilities configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra settings configs
│   ├── local                   <- Local configs
│   ├── logger                  <- Logger configs
│   ├── module                  <- Module configs
│   ├── paths                   <- Project paths configs
│   ├── trainer                 <- Trainer configs
│   │
│   ├── eval.yaml               <- Main config for evaluation
│   └── train.yaml              <- Main config for training
│
├── data                        <- Project data
├── logs                        <- Logs generated by hydra, lightning loggers, etc.
├── notebooks                   <- Jupyter notebooks.
├── scripts                     <- Shell scripts
│
├── src                         <- Source code
│   ├── callbacks               <- Additional callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── modules                 <- Lightning modules
│   ├── utils                   <- Utility scripts
│   │
│   ├── eval.py                 <- Run evaluation
│   └── train.py                <- Run training
│
├── tests                       <- Tests of any kind
│
├── .dockerignore               <- List of files ignored by docker
├── .gitattributes              <- List of additional attributes to pathnames
├── .gitignore                  <- List of files ignored by git
├── .pre-commit-config.yaml     <- Configuration of pre-commit hooks for code formatting
├── Dockerfile                  <- Dockerfile
├── Makefile                    <- Makefile with commands like `make train` or `make test`
├── pyproject.toml              <- Configuration options for testing and linting
├── requirements.txt            <- File for installing python dependencies
├── setup.py                    <- File for installing project as a package
└── README.md
```

## Data Preparation
### Structure
Our pipeline collect data as `.csv` files with a certain columns, which are defined in `src/datamodules/components/common.py` as:
```python
@dataclass(frozen=True)
class BaseDatasetCols:
    DATASET: Literal['dataset_name'] = 'dataset_name'
    LANGUAGE: Literal['language'] = 'language'
    NATIONALITY: Literal['country'] = 'country'
    SR: Literal['sample_rate'] = 'sample_rate'
    SPEAKER_ID: Literal['speaker_id'] = 'speaker_id'
    CLASS_ID: Literal['class_id'] = 'class_id'
    SPEAKER_NAME: Literal['speaker_name'] = 'speaker_name'
    GENDER: Literal['gender'] = 'gender'
    SPLIT: Literal['split'] = 'split'
    REC_DURATION: Literal['recording_duration'] = 'recording_duration'
    REL_FILEPATH: Literal['rel_filepath'] = 'rel_filepath'
    TEXT: Literal['text'] = 'text'
```
Additional columns can be added by overriding the base columns. Non-existing are set to defaults defined in `common.py`.

This enforced homogeneity in columns allows composing datasets without complications.

### Preprare the csvs
Follow `scripts/datasets/prep_{DATASET}.sh`. If you face any problems with these scripts, please report to ahmad.aloradi94@gmail.com.

#### Known Issues: 
1. `VoicePrivacy2025` dataset: when untarring the `T25-1` model's data, there is a mis-named . PLEASE FIX typo MANUALLY.
2. `LibriSpeech` dataset: In `SPEAKERS.TXT`, line 60 used to create a problem when loading as `.csv` with `sep='|'`. It is now automatically handleded.

### Recipes
At the moment we support recipes for the following datasets: `VoxCeleb`, `LibriSpeech`, `VoicePrivacy2025`. Currecntly, we expect the dataset to be downloaded on your machine, but we are slowly trying to intgrate the download in the `scripts/datasets`.