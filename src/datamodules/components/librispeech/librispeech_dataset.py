from dataclasses import dataclass

from src.datamodules.components.utils import BaseCollate, BaseDataset
from src.datamodules.components.common import get_dataset_class, LibriSpeechDefaults, DatasetItem

DATASET_DEFAULTS = LibriSpeechDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)

@dataclass
class LibrispeechItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0

class Collate(BaseCollate):
    pass

class LibrispeechDataset(BaseDataset):
    pass


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--librispeech_dir", type=str, default="data/librispeech",)
    parser.add_argument("--data_filepath", type=str,  default="data/librispeech/metadata/dev-clean.csv")
    args = parser.parse_args()

    librispeech_data = LibrispeechDataset(data_dir=args.librispeech_dir, data_filepath=args.data_filepath, 
                                          sample_rate=16000.0,)
    print("Number of samples: ", len(librispeech_data))
    
    # test dataset in a dataloder
    dataloder = DataLoader(librispeech_data, batch_size=2, shuffle=True, collate_fn=Collate())
    for batch in dataloder:
        print(batch)
        break
