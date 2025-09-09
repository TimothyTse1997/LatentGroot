import torch
import torch.nn as nn

import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForPreTraining

from audioseal.models import AudioSealDetector

class BaseAudioSealClassifier(AudioSealDetector):

    def __init__(
        self,
        model_kwargs = {
            "activation": "ELU",
            "activation_params": {"alpha": 1.0},
            "causal": False,
            "channels": 1,
            "compress": 2,
            "dilation_base": 2,
            "dimension": 128,
            "disable_norm_outer_blocks": 0,
            "kernel_size": 7,
            "last_kernel_size": 7,
            "lstm": 2,
            "n_filters": 32,
            "n_residual_layers": 1,
            "norm": "weight_norm",
            "norm_params": {},
            "pad_mode": "constant",
            "ratios": [8,5,4,2],,
            "residual_kernel_size": 3,
            "true_skip": True,
            "output_dim": 32,
            "nbits": 10,
        }
    ):
        super()._init__(**model_kwargs)
        self.loss_fn = nn.NLLLoss()
        pass
  
    def calculate_loss(
        self, 
        x: torch.Tensor,
        labels: torch.Tensor,
        sample_rate=None, # binary labels, (B,)
    ):
        logits, _ = self.forward(x, sample_rate=sample_rate)
        logits = torch.log(logits) # (B, 2)
        loss = self.loss_fn(logits, labels)
        return logits, loss

class SLIMEncoder(nn.Module):
    def __init__(self,
            input_dim=1024*2, output_dim=128,
            style_latent_layers=(0, 11),
            ling_latent_layers=(12, 22),
            **kwargs
        ):
        super().__init__()
        self.style_latent_layers = style_latent_layers
        self.ling_latent_layers = ling_latent_layers
        self.output_dim = output_dim
        self.style_encoder = Wav2Vec2ForPreTraining.from_pretrained(
                "r-f/wav2vec-english-speech-emotion-recognition")
        self.ling_encoder = Wav2Vec2ForCTC.from_pretrained(
                "jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.freeze_encoders()
        self.proj = nn.Linear(input_dim, output_dim)

    def freeze_encoders(self):
        for param in self.ling_encoder.parameters():
            param.requires_grad = False
        for param in self.style_encoder.parameters():
            param.requires_grad = False

    def _mean_latent(self, latents: list):
        return torch.stack(latents).permute(1, 0, 2, 3).mean(1)
    
    def checkpoint(self, checkpoint_path):
        # We ONLY save the projection layer
        torch.save(self.proj.state_dict(), checkpoint_path)
    
    def load_from_checkpoint(self, checkpoint_path):
        # We ONLY load the projection layer
        self.proj.load_state_dict(
            torch.load(checkpoint_path, weights_only=True)
        )

    def forward(self, input_values, attention_mask):
        # ling encoding list[(B, seq_len, H)]
        ling_latents = self.ling_encoder(input_values, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        ling_latent = self._mean_latent(
            ling_latents[
                self.ling_latent_layers[0]:
                self.ling_latent_layers[1]
            ])

        # style encoding list[(B, seq_len, H)]
        style_latents = self.style_encoder(input_values, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        style_latent = self._mean_latent(
            style_latents[
                self.style_latent_layers[0]:
                self.style_latent_layers[1]
            ])

        full_latent = torch.cat((style_latent, ling_latent), dim=-1) # B, seq_len, 2H
        proj_latent = self.proj(full_latent) # B, seq_len, output_dim
        return proj_latent

if __name__ == "__main__":
    slim_encoder = SLIMEncoder()
    slim_encoder.cuda()
    pass