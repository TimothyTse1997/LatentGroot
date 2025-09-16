import gc
import json
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import sys
import traceback

import torchaudio

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# from accelerate import Accelerator

from latent_encoder import (
    SLIMEncoder,
    BaseAudioSealClassifier,
    load_from_pretrain,
    HalfFreezeW2VClassifier,
)
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
        clip_length=8,
        fp16=False
        # gradient_accumulation_steps=5
    ):
        self.num_batch_logged = num_batch_logged
        self.num_max_train_steps = num_max_train_steps
        self.num_max_eval_steps = num_max_eval_steps
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

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
            clip_length=clip_length,
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

    def get_used_vram(self):
        device = torch.device(self.device)
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024**2
        return round(int(mem_used_MB))

    def log_batch(self, batch):
        inputs, labels = batch
        speech = inputs.input_values[0]
        save_dir = self.checkpoint_dir / "example_audio/"
        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / f"sample_audio_{self.logged_audio}.wav"

        torchaudio.save(save_path, speech.unsqueeze(0), self.sampling_rate)
        self.logged_audio += 1
        del speech
        pass

    def train_step(self, batch):
        self.optimizer.zero_grad()

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)
        loss = None
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

        del inputs, labels, logit, latent

        return loss.detach().cpu()

    def train_step_amp(self, batch):
        # print("using fp16")

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            with torch.no_grad():
                latent = self.encoder(
                    inputs.input_values.to(self.device),
                    inputs.attention_mask.to(self.device),
                )
            logit, loss = self.detector.calculate_loss(
                latent, labels, sample_rate=self.sampling_rate
            )

            # assert loss.dtype is torch.float32
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.optimizer.zero_grad()
        del latent, inputs, labels, logit

        return loss.detach().float().cpu()

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

    @torch.no_grad()
    def eval_step_amp(self, batch):
        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            latent = self.encoder(
                inputs.input_values.to(self.device),
                inputs.attention_mask.to(self.device),
            )

            logit, loss = self.detector.calculate_loss(
                latent, labels, sample_rate=self.sampling_rate
            )
        return logit.detach().float().cpu(), loss.detach().float().cpu()

    def train_epoch(self, epoch_num):
        self.detector.train(True)
        pbar = tqdm(
            range(len(self.train_dataloader)),
            desc="Training",
            dynamic_ncols=True,
            position=1,
        )
        total_loss = 0

        self.optimizer.zero_grad()
        for i, _ in enumerate(pbar):

            try:
                batch = next(iter(self.train_dataloader))

                if self.checkpoint_dir and (self.logged_audio < self.num_batch_logged):
                    self.log_batch(batch)
                if not self.fp16:
                    loss = self.train_step(batch)
                else:
                    loss = self.train_step_amp(batch)
                total_loss += float(loss.numpy())
                pbar.set_postfix(
                    {
                        f"avg_loss_epoch_{epoch_num}": total_loss / (i + 1),
                        "used_mem": f"{self.get_used_vram()}MB",
                    }
                )
                self.step += 1
                del batch, loss
                # gc.collect()
                # torch.cuda.empty_cache()

            # except BaseException as ex:
            except Exception as e:
                traceback.print_exc()
                # Get current system exception
                # ex_type, ex_value, ex_traceback = sys.exc_info()

                # Extract unformatter stack traces as tuples
                # trace_back = traceback.extract_tb(ex_traceback)

                # Format stacktrace
                # stack_trace = list()

                # for trace in trace_back:
                #    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

                # print("Exception type : %s " % ex_type.__name__)
                # print("Exception message : %s" %ex_value)
                # print("Stack trace : %s" %stack_trace)
                loss, batch = None, None
                gc.collect()
                torch.cuda.empty_cache()
                continue

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
            if not self.fp16:
                logit, loss = self.eval_step(batch)
            else:
                logit, loss = self.eval_step_amp(batch)

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
        clip_length=8,
        fp16=False
        # gradient_accumulation_steps=5
    ):
        self.num_batch_logged = num_batch_logged
        self.num_max_train_steps = num_max_train_steps
        self.num_max_eval_steps = num_max_eval_steps
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.num_epoch_per_checkpoint = num_epoch_per_checkpoint
        if not checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir()

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
            clip_length=clip_length,
            use_wav_feature=False,
        )

        self.detector = load_from_pretrain(pretrain_checkpoint_path)

        self.detector_kwargs = {"pretrain_checkpoint_path": pretrain_checkpoint_path}
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

    def log_batch(self, batch):
        inputs, labels = batch
        speech = inputs[0]
        save_dir = self.checkpoint_dir / "example_audio/"
        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / f"sample_audio_{self.logged_audio}.wav"

        torchaudio.save(save_path, speech.unsqueeze(0), self.sampling_rate)
        self.logged_audio += 1
        del speech
        pass

    def train_step(self, batch):
        self.optimizer.zero_grad()

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        logit, loss = self.detector.calculate_loss(
            inputs.to(self.device), labels, sample_rate=self.sampling_rate
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
            inputs.to(self.device), labels, sample_rate=self.sampling_rate
        )
        return logit.detach().cpu(), loss.detach().cpu()


class W2VTrainer(BaseTrainer):
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
        clip_length=8,
        fp16=False
        # gradient_accumulation_steps=5
    ):
        self.num_batch_logged = num_batch_logged
        self.num_max_train_steps = num_max_train_steps
        self.num_max_eval_steps = num_max_eval_steps
        self.fp16 = fp16
        if self.fp16:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.num_epoch_per_checkpoint = num_epoch_per_checkpoint
        if not checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir()

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
            clip_length=clip_length,
        )

        self.detector = HalfFreezeW2VClassifier()

        self.detector_kwargs = {}
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
            inputs.input_values.to(self.device),
            inputs.attention_mask.to(self.device),
            labels,
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach().cpu()

    def train_step_amp(self, batch):
        # print("using fp16")

        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16):

            logit, loss = self.detector.calculate_loss(
                inputs.input_values.to(self.device),
                inputs.attention_mask.to(self.device),
                labels,
            )

            # assert loss.dtype is torch.float32
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.optimizer.zero_grad()
        del inputs, labels, logit

        return loss.detach().float().cpu()

    @torch.no_grad()
    def eval_step(self, batch):
        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        logit, loss = self.detector.calculate_loss(
            inputs.input_values.to(self.device),
            inputs.attention_mask.to(self.device),
            labels,
        )
        return logit.detach().cpu(), loss.detach().cpu()

    @torch.no_grad()
    def eval_step_amp(self, batch):
        inputs, labels = batch
        labels = labels.type(torch.LongTensor).to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            logit, loss = self.detector.calculate_loss(
                inputs.input_values.to(self.device),
                inputs.attention_mask.to(self.device),
                labels,
            )

        return logit.detach().float().cpu(), loss.detach().float().cpu()


if __name__ == "__main__":
    data_paths = {
        # "wm_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_watermarked",
        "wm_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel_watermarked_fix_noise/watermarked",
        # "tts_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train",
        "tts_data_dir": "/home/tst000/projects/datasets/LibriTTS_synthesize/train_parallel_watermarked_fix_noise/original",
        # "raw_audio_dir": "/home/tst000/projects/datasets/LibriTTS/train-clean-100",
        "raw_audio_dir": "",
    }
    trainer = RawAudioTrainer(
        # trainer = BaseTrainer(
        data_paths=data_paths,
        batch_size=64,
        checkpoint_dir="/home/tst000/projects/checkpoint/non_latent_groot_trial0/",
        # num_max_train_steps=15,
        # num_max_eval_steps=15,
        epochs=100,
        drop_raw_data=True,
        fp16=False,
        pretrain_checkpoint_path="/gpfs/fs5/nrc/nrc-fs1/ict/others/u/tst000/.cache/audioseal/94c8df0b1d5ea8e45af4c884",
    )
    trainer.fit()
