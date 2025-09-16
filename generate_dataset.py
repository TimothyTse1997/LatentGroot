import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import numpy as np
import torch

from datasets import load_dataset

from f5_tts.api import F5TTS


def save_jsonl(JSON_file, save_fname):
    with open(save_fname, "w") as outfile:
        for entry in JSON_file:
            json.dump(entry, outfile)
            outfile.write("\n")


class FixNoiseWatermark:
    def __init__(self, fix_noise=None, fix_noise_size=(100, 16000), **kwargs):
        self.fix_noise = fix_noise

        if self.fix_noise is None:
            self.fix_noise_size = fix_noise_size
            self.fix_noise = torch.randn(self.fix_noise_size)
        else:
            self.fix_noise_size = self.fix_noise.shape
        pass

    @torch.no_grad()
    def __call__(self, y0, **kwargs):
        batch_size, seq_len, num_channels = y0.shape
        assert num_channels == self.fix_noise_size[0]
        num_patch = (seq_len // self.fix_noise_size[-1]) + 1
        new_noise = self.fix_noise.clone().repeat(1, num_patch)[:, :seq_len]
        new_noise = (
            torch.stack([new_noise for _ in range(batch_size)])
            .to(y0.dtype)
            .to(y0.device)
        ).permute(0, 2, 1)
        assert new_noise.shape == y0.shape

        return new_noise


def watermark_fn(y0, watermark_code: list = []):
    assert len(watermark_code) == 10
    batch_size, num_channels, seq_len = y0.shape
    assert num_channels == 100
    watermark_values = [-0.1 if wc == 0 else 0.1 for wc in watermark_code]
    full_watermark = torch.tensor(sum([watermark_values for _ in range(10)], [])).to(
        y0.device
    )
    y0 = (y0.permute(0, 2, 1) + full_watermark).permute(0, 2, 1)
    return y0


def get_ref_text_from_wav(wav_path):
    wav_path = wav_path.absolute()
    wav_parent, wav_name = wav_path.parent, wav_path.name
    wav_name = wav_name.split(".")[0]
    text_fname = wav_parent / f"{wav_name}.normalized.txt"
    with open(text_fname, "r") as f:
        ref_text = f.readline().replace("\n", "")
    return ref_text


def main(
    tts_dataset_path=Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/"),
    output_dir=Path("/home/tst000/projects/datasets/LibriTTS_synthesize/train/"),
    num_audio_per_speaker=40,
    noise_update_fn=None,
    fix_codec=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
):
    text_dataset = load_dataset(
        "agentlans/high-quality-english-sentences", split="test"
    )["text"]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    ref_audio_dict = {
        speaker_path.name: list(speaker_path.glob("**/*.wav"))[:num_audio_per_speaker]
        for speaker_path in tts_dataset_path.glob("*")
    }
    text_id = 0
    f5tts = F5TTS()
    meta_data = []

    for speaker, ref_files in tqdm(ref_audio_dict.items()):

        for i, ref_file in enumerate(ref_files):
            if noise_update_fn is not None:
                codec = (
                    fix_codec
                    if fix_codec is not None
                    else np.random.binomial(1, 0.5, 10).tolist()
                )
                wm_fn = partial(noise_update_fn, watermark_code=codec)
                f5tts.ema_model.noise_update_fn = noise_update_fn
            else:
                codec = None

            save_path = output_dir / f"spk_{speaker}_txt_{text_id}_ref_{i}.wav"
            ref_text = get_ref_text_from_wav(ref_file)
            wav_meta_data = {
                "ref_file": str(ref_file.absolute()),
                "ref_text": str(ref_text),
                "gen_text": str(text_dataset[text_id]),
                "file_wave": str(save_path.absolute()),
                "codec": codec,
            }

            _ = f5tts.infer(
                ref_file=str(ref_file),
                ref_text=ref_text,
                gen_text=str(text_dataset[text_id]),
                file_wave=save_path.absolute(),
                seed=None,
            )
            del _
            meta_data.append(wav_meta_data)
        text_id += 1


@torch.no_grad()
def main_parallel(
    tts_dataset_path=Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/"),
    output_dir=Path(
        "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel/"
    ),
    num_audio_per_speaker=200,
    noise_update_fn=None,
    fix_codec=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
):
    text_dataset = load_dataset(
        "agentlans/high-quality-english-sentences", split="test"
    )["text"]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    wm_dir = output_dir / "watermarked"
    if not wm_dir.exists():
        wm_dir.mkdir(parents=True)

    tts_dir = output_dir / "original"
    if not tts_dir.exists():
        tts_dir.mkdir(parents=True)

    ref_audio_dict = {
        speaker_path.name: list(speaker_path.glob("**/*.wav"))[:num_audio_per_speaker]
        for speaker_path in tts_dataset_path.glob("*")
    }
    text_id = 0
    f5tts = F5TTS()
    all_wm_meta_data = []
    all_tts_meta_data = []

    for speaker, ref_files in tqdm(ref_audio_dict.items(), position=0):

        # for i, ref_file in enumerate(ref_files):
        for i, audio_id in enumerate(tqdm(range(num_audio_per_speaker), position=1)):
            ref_file_id = audio_id % len(ref_files)
            ref_file = ref_files[ref_file_id]
            text = str(text_dataset[text_id])

            f5tts.ema_model.noise_update_fn = None

            tts_save_path = tts_dir / f"spk_{speaker}_txt_{text_id}_ref_{i}.wav"
            ref_text = get_ref_text_from_wav(ref_file)

            tts_meta_data = {
                "ref_file": str(ref_file.absolute()),
                "ref_text": str(ref_text),
                "gen_text": text,
                "file_wave": str(tts_save_path.absolute()),
            }

            _ = f5tts.infer(
                ref_file=str(ref_file),
                ref_text=ref_text,
                gen_text=text,
                file_wave=tts_save_path.absolute(),
                seed=None,
            )
            del _
            all_tts_meta_data.append(tts_meta_data)

            if noise_update_fn is not None:
                codec = (
                    fix_codec
                    if fix_codec is not None
                    else np.random.binomial(1, 0.5, 10).tolist()
                )
                wm_fn = partial(noise_update_fn, watermark_code=codec)
                f5tts.ema_model.noise_update_fn = noise_update_fn
            else:
                codec = None

            wm_save_path = wm_dir / f"spk_{speaker}_txt_{text_id}_ref_{i}.wav"
            ref_text = get_ref_text_from_wav(ref_file)

            wm_meta_data = {
                "ref_file": str(ref_file.absolute()),
                "ref_text": str(ref_text),
                "gen_text": text,
                "file_wave": str(wm_save_path.absolute()),
                "codec": codec,
            }

            _ = f5tts.infer(
                ref_file=str(ref_file),
                ref_text=ref_text,
                gen_text=text,
                file_wave=wm_save_path.absolute(),
                seed=None,
            )
            del _
            all_wm_meta_data.append(wm_meta_data)
            text_id += 1

    save_jsonl(all_tts_meta_data, output_dir / "tts_meta_data.jsonl")
    save_jsonl(all_wm_meta_data, output_dir / "wm_meta_data.jsonl")


if __name__ == "__main__":
    # main()
    # main(
    #    tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/dev-clean/"),
    #    output_dir = Path("/home/tst000/projects/datasets/LibriTTS_synthesize/dev_watermarked_fix_noise/"),
    #    noise_update_fn=watermark_fn
    # )
    fix_noise_wm_fn = FixNoiseWatermark()
    main_parallel(
        tts_dataset_path=Path(
            "/home/tst000/projects/datasets/LibriTTS/train-clean-100/"
        ),
        output_dir=Path(
            "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel_watermarked_fix_noise/"
        ),
        noise_update_fn=fix_noise_wm_fn,
    )
