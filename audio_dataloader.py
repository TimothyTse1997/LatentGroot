import os
from tqdm import tqdm
import random
from pathlib import Path
import time
from collections import defaultdict

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
        batch["speech"], batch["aug_name"] = self.aug_fn(batch["speech"])

        batch["label"] = data_sect

        return batch


class WMCustomPathDataset(WMBinaryClassificationDataset):
    def __init__(
        self,
        wm_data_paths: list,
        tts_data_paths: list,
        sampling_rate=16000,
        clip_length=8,
        drop_raw_data=False,
        **kwargs,
    ):

        self.wm_data = wm_data_paths
        self.tts_data = tts_data_paths
        self.sampling_rate = sampling_rate

        assert len(self.wm_data) == len(self.tts_data)

        self.data_len = len(self.wm_data)

        wm_iter = self.wm_data

        non_wm_iter = self.tts_data

        self.balance_data_list = list(zip(non_wm_iter, wm_iter))
        self.data = list(zip(*self.balance_data_list))

        assert len(self.data[0]) == self.data_len
        assert len(self.data[1]) == self.data_len

        self.random_clip = RandomClip(
            self.sampling_rate, clip_length=self.sampling_rate * clip_length
        )
        self.aug_fn = RandomAudioAugmentation(sample_rate=self.sampling_rate)


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
        return audios, labels, augmentations


class W2VLabeledCollator(W2VBaseCollator):
    def __call__(self, batch):
        audios = [b["speech"] for b in batch]
        augmentations = [b["aug_name"] for b in batch]

        labels = torch.LongTensor([int(b["label"]) for b in batch])
        inputs = self.feature_extractor(
            audios, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )

        return inputs, labels, augmentations


def get_all_speaker_from_dir(data_dir):
    data_dir = Path(data_dir)
    all_wav_files = list(data_dir.glob("*.wav"))
    all_speaker = [int(fp.name.split("_")[1]) for fp in all_wav_files]
    all_speaker = set(all_speaker)
    return all_speaker


def get_speaker_wav_from_dir(speaker_id, data_dir):
    return list(Path(data_dir).glob(f"spk_{speaker_id}*.wav"))


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


def get_labeled_dataloader_manual_split(
    train_data_paths,
    eval_data_paths,
    sampling_rate=16000,
    eval_split=0.1,
    batch_size=16,
    drop_raw_data=False,
    clip_length=8,
    num_workers=16,
    use_wav_feature=True,
):
    train_dataset = WMBinaryClassificationDataset(
        sampling_rate=sampling_rate,
        drop_raw_data=drop_raw_data,
        clip_length=clip_length,
        **train_data_paths,
    )
    eval_dataset = WMBinaryClassificationDataset(
        sampling_rate=sampling_rate,
        drop_raw_data=drop_raw_data,
        clip_length=clip_length,
        **eval_data_paths,
    )

    if use_wav_feature:
        collate_fn = W2VLabeledCollator()
    else:
        collate_fn = RawAudioCollator()

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


def get_labeled_dataloader_split_by_speaker(
    data_paths,
    sampling_rate=16000,
    eval_split=0.1,
    batch_size=16,
    drop_raw_data=False,
    clip_length=8,
    num_workers=16,
    use_wav_feature=True,
):

    all_wm_speaker = get_all_speaker_from_dir(data_paths["wm_data_dir"])
    all_tts_speaker = get_all_speaker_from_dir(data_paths["tts_data_dir"])

    assert all_wm_speaker == all_tts_speaker

    all_wm_speaker = list(all_wm_speaker)
    all_tts_speaker = list(all_tts_speaker)

    num_speakers = len(all_tts_speaker)
    num_eval_speakers = max(1, int(num_speakers * eval_split))

    eval_speakers = random.sample(all_tts_speaker, num_eval_speakers)
    train_speakers = set(all_wm_speaker) - set(eval_speakers)

    if use_wav_feature:
        collate_fn = W2VLabeledCollator()
    else:
        collate_fn = RawAudioCollator()

    all_train_wm_path = sum(
        [
            get_speaker_wav_from_dir(speaker_id, data_paths["wm_data_dir"])
            for speaker_id in train_speakers
        ],
        [],
    )

    all_train_tts_path = sum(
        [
            get_speaker_wav_from_dir(speaker_id, data_paths["tts_data_dir"])
            for speaker_id in train_speakers
        ],
        [],
    )

    all_eval_wm_path = sum(
        [
            get_speaker_wav_from_dir(speaker_id, data_paths["wm_data_dir"])
            for speaker_id in eval_speakers
        ],
        [],
    )

    all_eval_tts_path = sum(
        [
            get_speaker_wav_from_dir(speaker_id, data_paths["tts_data_dir"])
            for speaker_id in eval_speakers
        ],
        [],
    )

    train_dataset = WMCustomPathDataset(
        wm_data_paths=all_train_wm_path,
        tts_data_paths=all_train_tts_path,
        sampling_rate=sampling_rate,
        drop_raw_data=drop_raw_data,
        clip_length=clip_length,
    )
    eval_dataset = WMCustomPathDataset(
        wm_data_paths=all_eval_wm_path,
        tts_data_paths=all_eval_tts_path,
        sampling_rate=sampling_rate,
        drop_raw_data=drop_raw_data,
        clip_length=clip_length,
    )

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


def create_directory(dir_path):
    if not dir_path.exists():
        dir_path.mkdir()
    return dir_path


def split_dataset(data_paths, data_split_dir="", eval_split=0.05):
    data_split_dir = Path(data_split_dir)
    if not data_split_dir.exists():
        data_split_dir.mkdir()

    eval_dir = create_directory(data_split_dir / "eval")
    train_dir = create_directory(data_split_dir / "train")

    wm_eval_dir = create_directory(eval_dir / "watermarked")
    tts_eval_dir = create_directory(eval_dir / "original")

    wm_train_dir = create_directory(train_dir / "watermarked")
    tts_train_dir = create_directory(train_dir / "original")

    wm_data_dir = Path(data_paths["wm_data_dir"])
    tts_data_dir = Path(data_paths["tts_data_dir"])
    all_valid_fname = []

    for wm_file in wm_data_dir.glob("*.wav"):
        wm_fname = wm_file.name
        if not (tts_data_dir / wm_fname).exists():
            continue
        all_valid_fname.append(wm_fname)

    num_eval_data = max(len(all_valid_fname) * eval_split, 1)

    all_wm_speaker = get_all_speaker_from_dir(data_paths["wm_data_dir"])
    data_per_speaker = max(int(num_eval_data // len(all_wm_speaker)), 1)

    num_eval_data = int(data_per_speaker * len(all_wm_speaker))

    print(
        f"number of evaluation data: {num_eval_data}, data_per_speaker: {data_per_speaker}"
    )
    all_speaker_dict = defaultdict(list)

    for fname in all_valid_fname:
        all_speaker_dict[int(fname.split("_")[1])].append(fname)

    all_sampled_data = []
    for speaker_id in all_wm_speaker:
        all_speaker_fname = all_speaker_dict[int(speaker_id)]
        sampled_data = random.sample(all_speaker_fname, data_per_speaker)
        all_sampled_data += sampled_data

    all_sampled_data = set(all_sampled_data)

    for fname in all_valid_fname:
        wm_fname = (wm_data_dir / fname).absolute()
        tts_fname = (tts_data_dir / fname).absolute()

        if fname in all_sampled_data:
            os.symlink(wm_fname, (wm_eval_dir / fname).absolute())
            os.symlink(tts_fname, (tts_eval_dir / fname).absolute())
            continue
        os.symlink(wm_fname, (wm_train_dir / fname).absolute())
        os.symlink(tts_fname, (tts_train_dir / fname).absolute())


if __name__ == "__main__":
    data_paths = {
        # "wm_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_watermarked",
        "wm_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel_watermarked_fix_noise/watermarked",
        # "tts_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train",
        "tts_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel_watermarked_fix_noise/original",
        # "raw_audio_dir": "/home/tst000/projects/datasets/LibriTTS/train-clean-100",
        "raw_audio_dir": "",
    }
    data_split_dir = "/home/tst000/projects/datasets/LibriTTS_synthesize/manual_splited_train_dataset_parallel"
    split_dataset(data_paths, data_split_dir=data_split_dir, eval_split=0.05)
    exit()

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
