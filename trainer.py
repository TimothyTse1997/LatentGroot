import torch
from torch.utils.data import Dataset, DataLoader, random_split

from latent_encoder import SLIMEncoder
from audio_dataloader import (
    BaseAudioDataset,
    W2VBaseCollator,
    WMBinaryClassificationDataset,
    W2VLabeledCollator
)

class BaseTrainer:
    def __init__(
        self,
        data_paths={
            "wm_data_dir": "",
            "tts_data_dir": "",
            "raw_audio_dir": ""
        },
        batch_size=16,
        sampling_rate=16000
    ):
        pass




if __name__ == "__main__":
    dummy_dataset = "/home/tst000/projects/LatentGroot/test/dummy_dataset"
    dataset = BaseAudioDataset(
        dataset_dir=dummy_dataset)
    collate_fn = W2VBaseCollator()
    debug_dataloader = DataLoader(
        dataset, batch_size=1,
        collate_fn=collate_fn, shuffle=True
    )
    batch = next(iter(debug_dataloader))
    print(batch)

    encoder = SLIMEncoder()
    encoder = encoder.cuda()

    _ = encoder.freeze_encoders()
    #batch = batch.cuda()

    with torch.no_grad():
        out = encoder(
            batch.input_values.cuda(),
            batch.attention_mask.cuda(),
        )

    print(out.shape)    
