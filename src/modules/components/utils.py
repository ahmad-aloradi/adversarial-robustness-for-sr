import torch
from speechbrain.pretrained import EncoderClassifier
import pandas as pd
from src.datamodules.components.utils import SimpleAudioDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict


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

