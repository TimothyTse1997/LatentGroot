from tqdm import tqdm
import random
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio

# import torchaudio.functional as taF

from transformers import Wav2Vec2FeatureExtractor

import librosa

from audio_augmentation import RandomClip, RandomAudioAugmentation


class BaseAudioDataset(Dataset):
    def __init__(self, dataset_dir="", sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.dataset_dir = Path(dataset_dir)

        self.data_paths = list(self.dataset_dir.glob("*.wav"))

    def __len__(self):
        return len(self.data_paths)

    def load_audio_to_batch(self, wav_path):
        speech_array, sampling_rate = librosa.load(wav_path, sr=self.sampling_rate)
        batch = {}
        batch["speech"] = speech_array
        return batch

    def __getitem__(self, index):
        wav_path = self.data_paths[index]
        batch = self.load_audio_to_batch(wav_path)
        return batch


def random_cycle(iterable):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    random.shuffle(iterable)
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
            yield element


class WMBinaryClassificationDataset(BaseAudioDataset):
    def __init__(
        self,
        wm_data_dir,
        tts_data_dir,
        raw_audio_dir,
        sampling_rate=16000,
        clip_length=8,
        drop_raw_data=False,
    ):
        self.wm_data_dir = Path(wm_data_dir)
        self.tts_data_dir = Path(tts_data_dir)

        self.sampling_rate = sampling_rate

        self.wm_data = self.get_audio_dir_paths(self.wm_data_dir)
        self.tts_data = self.get_audio_dir_paths(self.tts_data_dir)

        if not drop_raw_data:
            self.raw_audio_dir = Path(raw_audio_dir)
            self.raw_audio = self.get_audio_dir_paths(self.raw_audio_dir)

            self.data_len = max(
                len(self.wm_data),
                len(self.tts_data) + len(self.raw_audio),
            )
        else:
            self.data_len = max(len(self.wm_data), len(self.tts_data))

        wm_iter = self._update_data_iter(self.wm_data)

        if not drop_raw_data:
            non_wm_iter = self._update_data_iter(self.tts_data + self.raw_audio)
        else:
            non_wm_iter = self._update_data_iter(self.tts_data)

        self.balance_data_list = list(zip(non_wm_iter, wm_iter))
        self.data = list(zip(*self.balance_data_list))

        assert len(self.data[0]) == self.data_len
        assert len(self.data[1]) == self.data_len

        self.random_clip = RandomClip(
            self.sampling_rate, clip_length=self.sampling_rate * clip_length
        )
        self.aug_fn = RandomAudioAugmentation(sample_rate=self.sampling_rate)

    def _update_data_iter(self, data_iter):
        if len(data_iter) >= self.data_len:
            return data_iter
        data_iter = random_cycle(data_iter)
        return data_iter

    def __len__(self):
        return self.data_len * 2  # loop through all 3 list

    def get_audio_dir_paths(self, raw_audio_dir):
        return list(raw_audio_dir.glob("*.wav")) + list(raw_audio_dir.glob("**/*.wav"))

    def __getitem__(self, index):
        data_sect = index // self.data_len
        data_id = index % self.data_len
        # print(index, self.data_len, data_sect)
        # print(index, self.data_len, data_id)
        # print(len(self.data[data_sect]))

        wav_path = self.data[data_sect][data_id]
        batch = self.load_audio_to_batch(wav_path)

        batch["speech"] = self.random_clip(torch.from_numpy(batch["speech"])).numpy()
        batch["speech"] = self.aug_fn(batch["speech"])

        batch["label"] = data_sect

        return batch


class W2VBaseCollator:
    def __init__(
        self,
        model_id="r-f/wav2vec-english-speech-emotion-recognition",
        sampling_rate=16_000,
    ):
        self.model_id = model_id
        self.sampling_rate = sampling_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id, local_files_only=True
        )

    def __call__(self, batch):
        audios = [b["speech"] for b in batch]
        start = time.time()
        inputs = self.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        print(time.time() - start)
        return inputs


class RawAudioCollator:
    def __call__(self, batch):
        audios = pad_sequence(
            [torch.from_numpy(b["speech"]) for b in batch], batch_first=True
        ).float()
        labels = torch.LongTensor([int(b["label"]) for b in batch])
        return audios, labels


class W2VLabeledCollator(W2VBaseCollator):
    def __call__(self, batch):
        audios = [b["speech"] for b in batch]

        labels = torch.LongTensor([int(b["label"]) for b in batch])
        inputs = self.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )

        return inputs, labels


def get_labeled_dataloader(
    data_paths,
    sampling_rate=16000,
    eval_split=0.1,
    batch_size=16,
    drop_raw_data=False,
    clip_length=8,
    num_workers=16,
    use_wav_feature=True,
):
    dataset = WMBinaryClassificationDataset(
        sampling_rate=sampling_rate,
        drop_raw_data=drop_raw_data,
        clip_length=clip_length,
        **data_paths,
    )
    dataset_size = len(dataset)
    eval_size = max(int(dataset_size * eval_split), 1)
    train_size = dataset_size - eval_size

    num_workers = min(batch_size, num_workers)
    if use_wav_feature:
        collate_fn = W2VLabeledCollator()
    else:
        collate_fn = RawAudioCollator()

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,  # num_workers=num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # num_workers=num_workers
    )
    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    dummy_dataset = "/home/tst000/projects/LatentGroot/test/dummy_dataset"
    dummy_tts_dataset = "/home/tst000/projects/LatentGroot/test/dummy_wm_dataset"
    dummy_raw_dataset = "/home/tst000/projects/LatentGroot/test/dummy_wm_dataset"

    # dataset = BaseAudioDataset(dataset_dir=dummy_dataset)
    wm_dataset = WMBinaryClassificationDataset(
        wm_data_dir=dummy_dataset,
        tts_data_dir=dummy_tts_dataset,
        raw_audio_dir=dummy_raw_dataset,
        drop_raw_data=True,
        clip_length=5,
    )

    # collate_fn = RawAudioCollator()
    collate_fn = W2VLabeledCollator()
    debug_dataloader = DataLoader(
        wm_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True
    )
    for i in tqdm(range(100)):
        inputs, labels = next(iter(debug_dataloader))
        print(inputs)
        # print(inputs.attention_mask.shape)
        # print(labels.shape)
