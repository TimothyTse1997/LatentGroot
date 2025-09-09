import json
from pathlib import Path
from tqdm import tqdm
from functools import partial

import numpy as np

from datasets import load_dataset

from f5_tts.api import F5TTS

def watermark_fn(y0, watermark_code:list = []):
    assert(len(watermark_code) == 10)
    batch_size, num_channels, seq_len = y0.shape
    assert(num_channels == 100)
    watermark_values = [-0.1 if wc == 0 else 0.1 for wc in watermark_code]
    full_watermark = torch.tensor(sum([watermark_values for _ in range(10)], [])).to(y0.device)
    y0 = (y0.permute(0, 2, 1) + full_watermark).permute(0, 2, 1) 
    return y0

def get_ref_text_from_wav(wav_path):
    wav_path = wav_path.absolute()
    wav_parent, wav_name = wav_path.parent, wav_path.name
    wav_name = wav_name.split(".")[0]
    text_fname = wav_parent / f"{wav_name}.normalized.txt"
    with open(text_fname, 'r') as f:
        ref_text = f.readline().replace("\n", "")
    return ref_text

def main(
    tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/"),
    output_dir = Path("/home/tst000/projects/datasets/LibriTTS_synthesize/train/"),
    num_audio_per_speaker=40,
    noise_update_fn=None,
    fix_codec=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
):
    text_dataset = load_dataset("agentlans/high-quality-english-sentences", split="test")["text"]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    ref_audio_dict = {
        speaker_path.name: list(speaker_path.glob("**/*.wav"))[:num_audio_per_speaker] for speaker_path in tts_dataset_path.glob("*")
    }
    text_id = 0
    f5tts = F5TTS()
    meta_data = []

    for speaker, ref_files in tqdm(ref_audio_dict.items()):

        for i, ref_file in enumerate(ref_files):
            if noise_update_fn is not None:
                codec = fix_codec if fix_codec is not None else np.random.binomial(1, 0.5, 10).tolist()
                wm_fn = partial(noise_update_fn, watermark_code=codec)
                f5tts.noise_update_fn = noise_update_fn
            else:
                codec = None

            save_path = output_dir / f"spk_{speaker}_txt_{text_id}_ref_{i}.wav"
            ref_text = get_ref_text_from_wav(ref_file)
            wav_meta_data = {
                "ref_file": str(ref_file.absolute()),
                "ref_text": str(ref_text),
                "gen_text": str(text_dataset[text_id]),
                "file_wave": str(save_path.absolute()),
                "codec": codec
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

if __name__ == "__main__":
    #main()
    main(
        tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/dev-clean/"),
        output_dir = Path("/home/tst000/projects/datasets/LibriTTS_synthesize/dev_watermarked/"),
        noise_update_fn=watermark_fn
    )
    main(
        tts_dataset_path = Path("/home/tst000/projects/datasets/LibriTTS/train-clean-100/"),
        output_dir = Path("/home/tst000/projects/datasets/LibriTTS_synthesize/train_watermarked/"),
        noise_update_fn=watermark_fn
    )