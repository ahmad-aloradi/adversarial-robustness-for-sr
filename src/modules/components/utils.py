from typing import Optional, Dict
from collections import OrderedDict
import threading

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from speechbrain.inference.classifiers import EncoderClassifier

from src.datamodules.components.utils import SimpleAudioDataset
from src import utils

logger = utils.get_pylogger(__name__)


class EmbeddingCache:
    """Manages caching of text embeddings with LRU eviction policy and usage statistics.

    Attributes:
        max_size (int): Maximum number of embeddings the cache can hold.
    """

    def __init__(self, max_size: int = 10000):
        """Initializes the cache with a specified maximum size.

        Args:
            max_size (int): Positive integer indicating the maximum cache capacity.
        
        Raises:
            ValueError: If `max_size` is not a positive integer.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieves the embedding for a key, updating usage statistics.

        Args:
            key (str): The identifier for the embedding.

        Returns:
            Optional[torch.Tensor]: The cached tensor if exists, otherwise None.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            
            self.misses += 1
            return None

    def update(self, key: str, value: torch.Tensor) -> None:
        """Adds or updates an embedding in the cache.

        Args:
            key (str): The identifier for the embedding.
            value (torch.Tensor): The tensor to cache. Detached and moved to CPU.

        Raises:
            TypeError: If `value` is not a `torch.Tensor`.
        """
        with self._lock:
            if not isinstance(value, torch.Tensor):
                raise TypeError("Value must be a torch.Tensor")

            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)

            value = value.detach().cpu()
            self._cache[key] = value

    def stats(self) -> Dict[str, float]:
        """Get cache performance statistics.

        Returns:
            Dictionary containing:
            - hits: Number of successful cache retrievals
            - misses: Number of failed cache lookups
            - hit_rate: Ratio of hits to total lookups (0.0-1.0)
        """
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0
            }

    def reset_stats(self) -> None:
        """Reset performance statistics counters."""
        with self._lock:
            self.hits = 0
            self.misses = 0

    def clear(self) -> None:
        """Removes all entries from the cache while preserving statistics."""
        with self._lock:
            self._cache.clear()

    def resize(self, new_max_size: int) -> None:
        """Adjusts the cache capacity, evicting LRU entries if necessary."""
        with self._lock:
            if new_max_size <= 0:
                raise ValueError("new_max_size must be a positive integer")
            self.max_size = new_max_size
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: str) -> torch.Tensor:
        with self._lock:
            value = self.get(key)
            if value is None:
                raise KeyError(f"Key '{key}' not found in cache")
            return value

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.update(key, value)

    def __repr__(self) -> str:
        with self._lock:
            stats = self.stats()
            return (
                f"EmbeddingCache(max_size={self.max_size}, current_size={len(self)}, "
                f"hit_rate={stats['hit_rate']:.2f})"
            )

class LanguagePredictionModel():
    """
    Identifies the language of an audio file using a pre-trained model.
    """
    def __init__(self, wav_dir: str, crop_len: float = 8.0):
        self.wav_dir = wav_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop_len = crop_len

        # https://huggingface.co/speechbrain/lang-id-commonlanguage_ecapa
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-commonlanguage_ecapa",
            savedir="local/pretrained_models/lang-id-commonlanguage_ecapa",
            run_opts={"device": self.device}
        )
        
        # Dictionary mapping language codes to full names
        self.LANGUAGES = {
            'Arabic': 'ar',
            'Basque': 'eu',
            'Breton': 'br',
            'Catalan': 'ca',
            'Chinese_China': 'zh_cn',
            'Chinese_Hongkong': 'zh_hk',
            'Chinese_Taiwan': 'zh_tw',
            'Chuvash': 'cv',
            'Czech': 'cs',
            'Dhivehi': 'dv',
            'Dutch': 'nl',
            'English': 'en',
            'Esperanto': 'eo',
            'Estonian': 'et',
            'French': 'fr',
            'Frisian': 'fy',
            'Georgian': 'ka',
            'German': 'de',
            'Greek': 'el',
            'Hakha_Chin': 'cnh',
            'Indonesian': 'id',
            'Interlingua': 'ia',
            'Italian': 'it',
            'Japanese': 'ja',
            'Kabyle': 'kab',
            'Kinyarwanda': 'rw',
            'Kyrgyz': 'ky',
            'Latvian': 'lv',
            'Maltese': 'mt',
            'Mongolian': 'mn',
            'Persian': 'fa',
            'Polish': 'pl',
            'Portuguese': 'pt',
            'Romanian': 'ro',
            'Romansh_Sursilvan': 'rm_sursilv',
            'Russian': 'ru',
            'Sakha': 'sah',
            'Slovenian': 'sl',
            'Spanish': 'es',
            'Swedish': 'sv',
            'Tamil': 'ta',
            'Tatar': 'tt',
            'Turkish': 'tr',
            'Ukrainian': 'uk',
            'Welsh': 'cy'
            }
        # Dictionary mapping languages to assumed primary countries
        self.COUNTRIES = {
            'ar': 'sa',  # Saudi Arabia
            'eu': 'es',  # Spain (Basque region)
            'br': 'fr',  # France (Brittany)
            'ca': 'es',  # Spain (Catalonia)
            'zh_cn': 'cn',  # China
            'zh_hk': 'hk',  # Hong Kong
            'zh_tw': 'tw',  # Taiwan
            'cv': 'ru',  # Russia (Chuvashia)
            'cs': 'cz',  # Czech Republic
            'dv': 'mv',  # Maldives
            'nl': 'nl',  # Netherlands
            'en': 'us',  # United States
            'eo': 'int',  # International (Esperanto)
            'et': 'ee',  # Estonia
            'fr': 'fr',  # France
            'fy': 'nl',  # Netherlands (Friesland)
            'ka': 'ge',  # Georgia
            'de': 'de',  # Germany
            'el': 'gr',  # Greece
            'cnh': 'mm',  # Myanmar
            'id': 'id',  # Indonesia
            'ia': 'int',  # International (Interlingua)
            'it': 'it',  # Italy
            'ja': 'jp',  # Japan
            'kab': 'dz',  # Algeria (Kabylie)
            'rw': 'rw',  # Rwanda
            'ky': 'kg',  # Kyrgyzstan
            'lv': 'lv',  # Latvia
            'mt': 'mt',  # Malta
            'mn': 'mn',  # Mongolia
            'fa': 'ir',  # Iran
            'pl': 'pl',  # Poland
            'pt': 'pt',  # Portugal
            'ro': 'ro',  # Romania
            'rm_sursilv': 'ch',  # Switzerland
            'ru': 'ru',  # Russia
            'sah': 'ru',  # Russia (Sakha Republic)
            'sl': 'si',  # Slovenia
            'es': 'es',  # Spain
            'sv': 'se',  # Sweden
            'ta': 'in',  # India
            'tt': 'ru',  # Russia (Tatarstan)
            'tr': 'tr',  # Turkey
            'uk': 'ua',  # Ukraine
            'cy': 'gb'   # United Kingdom (Wales)
            }
    
    def preprare_dataloader(self, df: pd.DataFrame, cfg: Dict) -> DataLoader:
        dataset = SimpleAudioDataset(df, self.wav_dir, crop_len=self.crop_len)
        dataloader = DataLoader(dataset=dataset, collate_fn=SimpleAudioDataset.collate_fn, **cfg)
        return dataloader

    def forward(self, df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
        dataloader = self.preprare_dataloader(df, cfg=cfg)
        self.model.to(self.device)

        with torch.no_grad():
            for batch_waveforms, batch_indices in tqdm(dataloader, desc="Predicting language from audio files"):
                predicted_languages = self.model.classify_batch(batch_waveforms.to(self.device))[-1]
                for idx, pred_lang in zip(batch_indices, predicted_languages):
                    df.loc[idx, 'language'] = self.LANGUAGES[pred_lang]
                    df.loc[idx, 'country'] = self.COUNTRIES[self.LANGUAGES[pred_lang]]
        return df

