import gc
import json
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import torchaudio

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# from accelerate import Accelerator

from latent_encoder import SLIMEncoder, BaseAudioSealClassifier, load_from_pretrain
from audio_dataloader import get_labeled_dataloader
from metric import create_metrics
from optimizer import build_optimizer


class BaseTrainer:
    def __init__(
        self,
        data_paths={"wm_data_dir": "", "tts_data_dir": "", "raw_audio_dir": ""},
        detector_kwargs={},
        batch_size=16,
        sampling_rate=16000,
        eval_split=0.1,
        epochs=10,
        metrics=["F1", "precision"],
        metric_kwargs={},
        scheduler_params_dict={},
        device="cuda",
        checkpoint_dir="",
        num_epoch_per_checkpoint=2,
        num_max_train_steps=None,
        num_max_eval_steps=None,
        num_batch_logged=10,
        drop_raw_data=False,
        # gradient_accumulation_steps=5
    ):
        self.num_batch_logged = num_batch_logged
        self.num_max_train_steps = num_max_train_steps
        self.num_max_eval_steps = num_max_eval_steps

        self.num_epoch_per_checkpoint = num_epoch_per_checkpoint
        if not checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        self.epochs = epochs
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.device = device
        self.eval_split = eval_split

        self.train_dataloader, self.eval_dataloader = get_labeled_dataloader(
            data_paths,
            sampling_rate=self.sampling_rate,
            eval_split=self.eval_split,
            batch_size=self.batch_size,
            drop_raw_data=drop_raw_data,
        )

        self.detector_kwargs = detector_kwargs
        self.detector = BaseAudioSealClassifier(**detector_kwargs)
        _ = self.detector.to(self.device)

        scheduler_params_dict.update(
            {"epochs": self.epochs, "steps_per_epoch": len(self.train_dataloader)}
        )
        self.optimizer, self.scheduler = build_optimizer(
            self.detector, scheduler_params_dict
        )

        self.metric_fn = create_metrics(metric_names=metrics, **metric_kwargs)
        self.encoder = SLIMEncoder()
        _ = self.encoder.freeze_encoders()
        _ = self.encoder.to(self.device)
        # self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        # self.train_dataloader, self.detector, self.optimizer, self.scheduler = self.accelerator.prepare(
        #     self.train_dataloader, self.detector,
        #     self.optimizer, self.scheduler)

        self.step = 0
        self.logged_audio = 0
        pass

    def log_batch(self, batch):
        inputs, labels = batch
        speech = inputs.input_values[0]
        save_dir = self.checkpoint_dir / "example_audio/"
        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / f"sample_audio_{self.logged_audio}.wav"

        torchaudio.save(save_path, speech.unsqueeze(0), self.sampling_rate)
        self.logged_audio += 1
        pass

    def train_step(self, batch):
        self.optimizer.zero_grad()

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)
        with torch.no_grad():
            latent = self.encoder(
                inputs.input_values.to(self.device),
                inputs.attention_mask.to(self.device),
            )

        logit, loss = self.detector.calculate_loss(
            latent, labels, sample_rate=self.sampling_rate
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().cpu()

    @torch.no_grad()
    def eval_step(self, batch):
        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)
        latent = self.encoder(
            inputs.input_values.to(self.device),
            inputs.attention_mask.to(self.device),
        )

        logit, loss = self.detector.calculate_loss(
            latent, labels, sample_rate=self.sampling_rate
        )
        return logit.detach().cpu(), loss.detach().cpu()

    def train_epoch(self, epoch_num):
        self.detector.train(True)
        pbar = tqdm(
            range(len(self.train_dataloader)),
            desc="Training",
            dynamic_ncols=True,
            position=1,
        )
        total_loss = 0

        for i, _ in enumerate(pbar):

            try:
                batch = next(iter(self.train_dataloader))

                if self.checkpoint_dir and (self.logged_audio < self.num_batch_logged):
                    self.log_batch(batch)

                loss = self.train_step(batch)
                total_loss += float(loss.numpy())
                pbar.set_postfix({f"avg_loss_epoch_{epoch_num}": total_loss / (i + 1)})
                self.step += 1
            except Exception as e:
                print(f"error in step: {self.step}: {e}")
                loss, batch = None, None
                gc.collect()
                torch.cuda.empty_cache()

            if (self.num_max_train_steps is not None) and (
                self.step > self.num_max_train_steps
            ):
                break

        return round(float(total_loss / (i + 1)), 3)

    @torch.no_grad()
    def eval_epoch(self, epoch_num):
        self.detector.eval()
        pbar = tqdm(
            range(len(self.eval_dataloader)),
            desc="Eval",
            dynamic_ncols=True,
            position=2,
        )
        total_loss = 0
        metric_dict = defaultdict(float)

        for i, _ in enumerate(pbar):
            batch = next(iter(self.eval_dataloader))
            logit, loss = self.eval_step(batch)

            total_loss += float(loss.numpy())

            pbar.set_postfix({f"avg_loss_epoch_{epoch_num}": total_loss / (i + 1)})
            batch_metric = self.metric_fn(predictions=logit, labels=batch[-1])
            for k, v in batch_metric.items():
                metric_dict[k] += v
            avg_metric_dict = {
                k + f"_epoch_{epoch_num}": v / (i + 1) for k, v in metric_dict.items()
            }
            pbar.set_postfix(avg_metric_dict)
            if (self.num_max_eval_steps is not None) and (i > self.num_max_eval_steps):
                break

        return round(float(total_loss / (i + 1)), 3)

    def save_checkpoint(self, suffix=""):
        steps = int(self.step)
        checkpoint_save_path = (
            self.checkpoint_dir / f"checkpoint_{steps}{suffix}_state_dict.ckpt"
        )
        config = self.detector_kwargs

        config_save_path = self.checkpoint_dir / f"model_config.json"
        if not config_save_path.exists():
            with open(config_save_path, "w") as f:
                json.dump(config, f, indent=4)

        torch.save(self.detector.state_dict(), checkpoint_save_path)

    def load_checkpoint(self, pretrained_checkpoint_path):
        self.detector.load_state_dict(
            torch.load(pretrained_checkpoint_path, weight_only=True)
        )

    def fit(self):
        self.step = 0
        for epoch_num in tqdm(range(self.epochs), desc="Epoch", position=0):
            avg_train_loss = self.train_epoch(epoch_num)
            avg_eval_loss = self.eval_epoch(epoch_num)
            if not self.checkpoint_dir:
                continue
            if (epoch_num + 1) % self.num_epoch_per_checkpoint == 0:
                checkpoint_suffix = f"_tl_{avg_train_loss}_el_{avg_eval_loss}_"
                self.save_checkpoint(suffix=checkpoint_suffix)
            if (self.num_max_train_steps is not None) and (
                self.step > self.num_max_train_steps
            ):
                break

        if self.checkpoint_dir:
            checkpoint_suffix = "_last_checkpoint_"
            self.save_checkpoint(suffix=checkpoint_suffix)
        pass


class RawAudioTrainer(BaseTrainer):
    # Instead of using encoder and use latent for classification
    # we just use the audioseal pretrain detector
    def __init__(
        self,
        data_paths={"wm_data_dir": "", "tts_data_dir": "", "raw_audio_dir": ""},
        batch_size=16,
        sampling_rate=16000,
        eval_split=0.1,
        epochs=10,
        metrics=["F1", "precision"],
        metric_kwargs={},
        scheduler_params_dict={},
        device="cuda",
        checkpoint_dir="",
        num_epoch_per_checkpoint=2,
        num_max_train_steps=None,
        num_max_eval_steps=None,
        num_batch_logged=10,
        pretrain_checkpoint_path="",
        drop_raw_data=False,
        # gradient_accumulation_steps=5
    ):
        self.num_batch_logged = num_batch_logged
        self.num_max_train_steps = num_max_train_steps
        self.num_max_eval_steps = num_max_eval_steps

        self.num_epoch_per_checkpoint = num_epoch_per_checkpoint
        if not checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)

        self.epochs = epochs
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.device = device
        self.eval_split = eval_split

        self.train_dataloader, self.eval_dataloader = get_labeled_dataloader(
            data_paths,
            sampling_rate=self.sampling_rate,
            eval_split=self.eval_split,
            batch_size=self.batch_size,
            drop_raw_data=drop_raw_data,
        )

        self.detector = load_from_pretrain(pretrain_checkpoint_path)
        _ = self.detector.to(self.device)

        scheduler_params_dict.update(
            {"epochs": self.epochs, "steps_per_epoch": len(self.train_dataloader)}
        )
        self.optimizer, self.scheduler = build_optimizer(
            self.detector, scheduler_params_dict
        )

        self.metric_fn = create_metrics(metric_names=metrics, **metric_kwargs)

        # self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        # self.train_dataloader, self.detector, self.optimizer, self.scheduler = self.accelerator.prepare(
        #     self.train_dataloader, self.detector,
        #     self.optimizer, self.scheduler)

        self.step = 0
        self.logged_audio = 0
        pass

    def train_step(self, batch):
        self.optimizer.zero_grad()

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        logit, loss = self.detector.calculate_loss(
            inputs.input_values.to(self.device), labels, sample_rate=self.sampling_rate
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().cpu()

    @torch.no_grad()
    def eval_step(self, batch):
        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        logit, loss = self.detector.calculate_loss(
            inputs.input_values.to(self.device), labels, sample_rate=self.sampling_rate
        )
        return logit.detach().cpu(), loss.detach().cpu()


if __name__ == "__main__":
    data_paths = {
        "wm_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_watermarked",
        "tts_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train",
        "raw_audio_dir": "/home/tst000/projects/datasets/LibriTTS/train-clean-100",
    }
    trainer = BaseTrainer(
        data_paths=data_paths,
        batch_size=128,
        checkpoint_dir="/home/tst000/projects/checkpoint/latent_groot/",
        # num_max_train_steps=15,
        # num_max_eval_steps=15,
        epochs=20,
        drop_raw_data=True,
    )
    trainer.fit()
