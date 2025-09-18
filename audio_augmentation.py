from pathlib import Path
import random

import numpy as np

import torch
import torchaudio
import torchaudio.functional as F

import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d
import warnings
from typing import Union, Optional, Tuple, Dict, Any


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset : (offset + self.clip_length)]

        return self.vad(audio_data)  # remove silences at the beggining/end


class RandomAudioAugmentation:
    def __init__(
        self,
        num_aug_per_sample=1,
        aug_dict={
            # "gaussian_noise": [{"snr_db": 15}, {"snr_db": 20}, {"snr_db": 30}],
            "gaussian_noise": [{"snr_db": 30}],
            # "time_stretch": [{"speed_factor": 1.05}, {"speed_factor": 0.95}],
            # "lowpass_filter": [{"cutoff_ratio": 0.3}, {"cutoff_ratio": 0.5}],
            "lowpass_filter": [{"cutoff_ratio": 0.5}],
            # "highpass_filter": [{"cutoff_ratio": 0.1}, {"cutoff_ratio": 0.2}],
            "highpass_filter": [{"cutoff_ratio": 0.2}],
            "mp3_compression": [{"bitrate_kbps": 16}],
            "echo": [
                # {"delay_seconds": 0.15, "decay": 0.15},
                {"delay_seconds": 0.1, "decay": 0.1},
            ],
            "default": [{}],
        },
        min_length=3,
        clip_range=[0.9, 1.0],
        sample_rate=16000,
    ):
        self.sample_rate = sample_rate
        self.num_aug_per_sample = num_aug_per_sample
        self.aug_dict = aug_dict
        self.aug_names = list(self.aug_dict.keys())
        self.clip_range = clip_range
        self.perturber = AudioPerturbations(sample_rate=self.sample_rate)
        self.min_length = min_length * self.sample_rate

    def custom_clip(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length <= self.min_length:
            return audio_data

        clip_frac = random.uniform(*self.clip_range)
        assert clip_frac <= 1
        length = int(audio_length * clip_frac)

        cliper = RandomClip(self.sample_rate, self.sample_rate * length)
        return cliper(audio_data)

    def __call__(self, audio_data):
        # TODO make it a MULTIPLE augmentation pipeline
        # audio_data = self.custom_clip(audio_data).numpy()

        all_aug = []
        aug_name = random.choice(self.aug_names)
        aug_param = random.choice(self.aug_dict[aug_name])

        perturbed, aug_name = self.perturber.apply_perturbation(
            audio_data, aug_name, **aug_param
        )
        # return torch.from_numpy(perturbed)

        return perturbed, aug_name
        # for aug_name in random.sample(
        #    self.aug_names, self.num_aug_per_sample
        # ):
        #    all_aug.append((aug_name, random.choice(self.aug_dict[aug_name])))


class AudioPerturbations:
    """
    Implements SOTA audio perturbations to test watermark robustness.
    Based on AudioMarkBench (2024) and recent watermarking attack research.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize perturbation handler.

        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate

    def add_default(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        return audio

    # ===== SIGNAL PROCESSING PERTURBATIONS (AudioMarkBench) =====

    def add_gaussian_noise(self, audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add Gaussian noise at specified SNR.

        Args:
            audio: Input audio signal
            snr_db: Signal-to-noise ratio in dB (5-40 dB range)

        Returns:
            Audio with added noise
        """
        # Calculate signal power
        signal_power = np.mean(audio**2)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise

    def add_background_noise(
        self, audio: np.ndarray, snr_db: float = 20.0, noise_type: str = "pink"
    ) -> np.ndarray:
        """
        Add background noise (pink, brown, white) at specified SNR.

        Args:
            audio: Input audio signal
            snr_db: Signal-to-noise ratio in dB (5-40 dB range)
            noise_type: Type of background noise ("pink", "brown", "white")

        Returns:
            Audio with background noise
        """
        # Generate different types of noise
        if noise_type == "white":
            noise = np.random.normal(0, 1, len(audio))
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == "brown":
            noise = self._generate_brown_noise(len(audio))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Scale noise to desired SNR
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(signal_power / (noise_power * snr_linear))

        return audio + noise * noise_scale

    def apply_time_stretch(
        self, audio: np.ndarray, speed_factor: float = 1.1
    ) -> np.ndarray:
        """
        Apply time stretching (speed change without pitch change).

        Args:
            audio: Input audio signal
            speed_factor: Speed factor (0.7-1.5 range, >1 = faster)

        Returns:
            Time-stretched audio
        """
        # Use librosa for high-quality time stretching
        return librosa.effects.time_stretch(audio, rate=speed_factor)

    def apply_pitch_shift(self, audio: np.ndarray, n_steps: float = 1.0) -> np.ndarray:
        """
        Apply pitch shifting (±1 semitone typical).

        Args:
            audio: Input audio signal
            n_steps: Number of semitones to shift (±1.0 typical)

        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def apply_highpass_filter(
        self, audio: np.ndarray, cutoff_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Apply high-pass filter.

        Args:
            audio: Input audio signal
            cutoff_ratio: Cutoff frequency as ratio of Nyquist (0.1-0.5)

        Returns:
            High-pass filtered audio
        """
        nyquist = self.sample_rate / 2
        cutoff = cutoff_ratio * nyquist

        # Design high-pass filter
        sos = signal.butter(5, cutoff, btype="high", fs=self.sample_rate, output="sos")
        return signal.sosfilt(sos, audio)

    def apply_lowpass_filter(
        self, audio: np.ndarray, cutoff_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Apply low-pass filter.

        Args:
            audio: Input audio signal
            cutoff_ratio: Cutoff frequency as ratio of Nyquist (0.1-0.5)

        Returns:
            Low-pass filtered audio
        """
        nyquist = self.sample_rate / 2
        cutoff = cutoff_ratio * nyquist

        # Design low-pass filter
        sos = signal.butter(5, cutoff, btype="low", fs=self.sample_rate, output="sos")
        return signal.sosfilt(sos, audio)

    def apply_quantization(self, audio: np.ndarray, bit_depth: int = 8) -> np.ndarray:
        """
        Apply quantization (bit depth reduction).

        Args:
            audio: Input audio signal
            bit_depth: Target bit depth (4-64 bit range)

        Returns:
            Quantized audio
        """
        # Calculate quantization levels
        max_val = np.max(np.abs(audio))
        n_levels = 2**bit_depth
        step_size = 2 * max_val / n_levels

        # Quantize
        quantized = np.round(audio / step_size) * step_size
        return np.clip(quantized, -max_val, max_val)

    def apply_smoothing(self, audio: np.ndarray, window_size: int = 11) -> np.ndarray:
        """
        Apply smoothing filter.

        Args:
            audio: Input audio signal
            window_size: Smoothing window size (6-22 range)

        Returns:
            Smoothed audio
        """
        # Use moving average for smoothing
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size

        kernel = np.ones(window_size) / window_size
        return np.convolve(audio, kernel, mode="same")

    def add_echo(
        self, audio: np.ndarray, delay_seconds: float = 0.3, decay: float = 0.3
    ) -> np.ndarray:
        """
        Add echo effect.

        Args:
            audio: Input audio signal
            delay_seconds: Echo delay in seconds (0.1-0.9 range)
            decay: Echo decay factor (0.0-1.0)

        Returns:
            Audio with echo
        """
        delay_samples = int(delay_seconds * self.sample_rate)

        # Create echo
        echoed = audio.copy()
        if delay_samples < len(audio):
            echoed[delay_samples:] += audio[:-delay_samples] * decay

        return echoed

    def apply_reverberation(
        self, audio: np.ndarray, room_scale: float = 0.5, damping: float = 0.5
    ) -> np.ndarray:
        """
        Apply simple reverberation effect.

        Args:
            audio: Input audio signal
            room_scale: Room size scale (0.1-1.0)
            damping: Damping factor (0.1-1.0)

        Returns:
            Audio with reverberation
        """
        # Simple multi-tap delay for reverb simulation
        delays = [0.03, 0.05, 0.08, 0.13, 0.21]  # Multi-tap delays
        gains = [0.8, 0.6, 0.4, 0.3, 0.2]

        reverbed = audio.copy()

        for delay, gain in zip(delays, gains):
            delay_samples = int(delay * room_scale * self.sample_rate)
            if delay_samples < len(audio):
                reverbed[delay_samples:] += audio[:-delay_samples] * gain * damping

        return reverbed

    # ===== COMPRESSION ATTACKS =====

    def simulate_mp3_compression(
        self, audio: np.ndarray, bitrate_kbps: int = 32
    ) -> np.ndarray:
        """
        Simulate MP3 compression artifacts.

        Args:
            audio: Input audio signal
            bitrate_kbps: Target bitrate (8-40 kbps range)

        Returns:
            Audio with compression artifacts
        """
        # Simplified MP3 simulation using frequency domain processing
        # Real MP3 would require external encoder/decoder

        # Apply frequency domain compression simulation
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sample_rate)
        fft = np.fft.rfft(audio)

        # Simulate quantization in frequency domain
        compression_factor = bitrate_kbps / 128.0  # Normalize to 128 kbps baseline
        quantization_levels = int(256 * compression_factor)

        # Quantize magnitude
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        max_mag = np.max(magnitude)
        quantized_magnitude = (
            np.round(magnitude / max_mag * quantization_levels)
            / quantization_levels
            * max_mag
        )

        # Reconstruct
        compressed_fft = quantized_magnitude * np.exp(1j * phase)
        compressed_audio = np.fft.irfft(compressed_fft, len(audio))

        return compressed_audio

    # ===== ADVANCED PERTURBATIONS =====

    def apply_jittering(
        self, audio: np.ndarray, max_shift: int = 3, probability: float = 0.1
    ) -> np.ndarray:
        """
        Apply jittering attack (random time position changes).

        Args:
            audio: Input audio signal
            max_shift: Maximum sample shift
            probability: Probability of jittering each sample

        Returns:
            Jittered audio
        """
        jittered = audio.copy()

        for i in range(len(audio)):
            if random.random() < probability:
                shift = random.randint(-max_shift, max_shift)
                new_idx = max(0, min(len(audio) - 1, i + shift))
                jittered[i] = audio[new_idx]

        return jittered

    def apply_dynamic_range_compression(
        self, audio: np.ndarray, ratio: float = 4.0, threshold: float = -20.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression.

        Args:
            audio: Input audio signal
            ratio: Compression ratio
            threshold: Threshold in dB

        Returns:
            Compressed audio
        """
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)

        # Apply compression above threshold
        compressed_db = np.where(
            audio_db > threshold, threshold + (audio_db - threshold) / ratio, audio_db
        )

        # Convert back to linear
        compressed = np.sign(audio) * (10 ** (compressed_db / 20))
        return compressed

    def apply_equalization(
        self, audio: np.ndarray, eq_gains_db: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply multi-band equalization.

        Args:
            audio: Input audio signal
            eq_gains_db: EQ gains in dB for frequency bands

        Returns:
            Equalized audio
        """
        if eq_gains_db is None:
            # Default random EQ
            eq_gains_db = np.random.uniform(-6, 6, 6)

        # Define frequency bands (Hz)
        bands = [60, 200, 500, 1000, 3000, 8000]

        # Apply EQ
        equalized = audio.copy()

        for i, (freq, gain_db) in enumerate(zip(bands, eq_gains_db)):
            if i == 0:  # Low-pass for first band
                sos = signal.butter(
                    2, freq, btype="low", fs=self.sample_rate, output="sos"
                )
            elif i == len(bands) - 1:  # High-pass for last band
                sos = signal.butter(
                    2, freq, btype="high", fs=self.sample_rate, output="sos"
                )
            else:  # Band-pass for middle bands
                sos = signal.butter(
                    2,
                    [bands[i - 1], freq],
                    btype="band",
                    fs=self.sample_rate,
                    output="sos",
                )

            # Filter and apply gain
            band_audio = signal.sosfilt(sos, audio)
            gain_linear = 10 ** (gain_db / 20)
            equalized += band_audio * (gain_linear - 1)

        return equalized

    # ===== UTILITY METHODS =====

    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Generate white noise
        white = np.random.normal(0, 1, length)

        # Apply pink noise filter (approximation)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-8  # Avoid division by zero

        # 1/f spectrum
        pink_fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, length)

        return pink / np.std(pink)

    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum)."""
        # Generate white noise
        white = np.random.normal(0, 1, length)

        # Apply brown noise filter
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-8  # Avoid division by zero

        # 1/f^2 spectrum
        brown_fft = fft / freqs
        brown = np.fft.irfft(brown_fft, length)

        return brown / np.std(brown)

    # ===== PERTURBATION PIPELINE =====

    def get_all_perturbations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available perturbations with default parameters.

        Returns:
            Dictionary of perturbation names and their parameter ranges
        """
        return {
            # AudioMarkBench No-box perturbations
            "gaussian_noise": {"snr_db": [5, 10, 20, 30, 40]},
            "background_noise_pink": {"snr_db": [5, 10, 20, 30, 40]},
            "background_noise_white": {"snr_db": [5, 10, 20, 30, 40]},
            "time_stretch": {"speed_factor": [0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]},
            "pitch_shift": {"n_steps": [-1.0, -0.5, 0.5, 1.0]},
            "highpass_filter": {"cutoff_ratio": [0.1, 0.2, 0.3, 0.4, 0.5]},
            "lowpass_filter": {"cutoff_ratio": [0.1, 0.2, 0.3, 0.4, 0.5]},
            "quantization": {"bit_depth": [4, 6, 8, 12, 16]},
            "smoothing": {"window_size": [6, 11, 16, 22]},
            "echo": {"delay_seconds": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "reverberation": {
                "room_scale": [0.3, 0.5, 0.8],
                "damping": [0.3, 0.5, 0.8],
            },
            "mp3_compression": {"bitrate_kbps": [8, 16, 24, 32, 40]},
            # Additional perturbations
            "jittering": {"max_shift": [1, 2, 3], "probability": [0.05, 0.1, 0.2]},
            "dynamic_compression": {
                "ratio": [2.0, 4.0, 8.0],
                "threshold": [-30, -20, -10],
            },
            "equalization": {"random_eq": True},
        }

    def apply_perturbation(
        self, audio: np.ndarray, perturbation_name: str, **kwargs
    ) -> np.ndarray:
        """
        Apply a specific perturbation to audio.

        Args:
            audio: Input audio signal
            perturbation_name: Name of perturbation to apply
            **kwargs: Parameters for the perturbation

        Returns:
            Perturbed audio
        """
        perturbation_map = {
            "default": self.add_default,
            "gaussian_noise": self.add_gaussian_noise,
            "background_noise_pink": lambda x, **k: self.add_background_noise(
                x, noise_type="pink", **k
            ),
            "background_noise_white": lambda x, **k: self.add_background_noise(
                x, noise_type="white", **k
            ),
            "time_stretch": self.apply_time_stretch,
            "pitch_shift": self.apply_pitch_shift,
            "highpass_filter": self.apply_highpass_filter,
            "lowpass_filter": self.apply_lowpass_filter,
            "quantization": self.apply_quantization,
            "smoothing": self.apply_smoothing,
            "echo": self.add_echo,
            "reverberation": self.apply_reverberation,
            "mp3_compression": self.simulate_mp3_compression,
            "jittering": self.apply_jittering,
            "dynamic_compression": self.apply_dynamic_range_compression,
            "equalization": self.apply_equalization,
        }

        if perturbation_name not in perturbation_map:
            raise ValueError(f"Unknown perturbation: {perturbation_name}")

        try:
            return (
                perturbation_map[perturbation_name](audio, **kwargs),
                perturbation_name,
            )
        except Exception as e:
            warnings.warn(f"Failed to apply {perturbation_name}: {e}")
            return audio, "default"  # Return original audio if perturbation fails


def main(audio_file, save_dir):
    """Example usage of audio perturbations."""
    # import sys

    # if len(sys.argv) < 2:
    #   print("Usage: python perturbations.py <audio_file>")
    #   return

    # Load audio
    # audio_file = sys.argv[1]
    audio, sr = librosa.load(audio_file, sr=16000)

    # Initialize perturbations
    perturber = AudioPerturbations(sample_rate=sr)

    # Apply some example perturbations
    print(f"Original audio shape: {audio.shape}")

    # Test different perturbations
    perturbations_to_test = [
        ("gaussian_noise", {"snr_db": 5}),
        ("gaussian_noise", {"snr_db": 10}),
        ("gaussian_noise", {"snr_db": 20}),
        ("gaussian_noise", {"snr_db": 30}),
        ("time_stretch", {"speed_factor": 1.2}),
        ("time_stretch", {"speed_factor": 1.1}),
        ("time_stretch", {"speed_factor": 0.9}),
        ("time_stretch", {"speed_factor": 0.8}),
        ("lowpass_filter", {"cutoff_ratio": 0.1}),
        ("lowpass_filter", {"cutoff_ratio": 0.3}),
        ("lowpass_filter", {"cutoff_ratio": 0.5}),
        ("highpass_filter", {"cutoff_ratio": 0.1}),
        ("highpass_filter", {"cutoff_ratio": 0.3}),
        ("highpass_filter", {"cutoff_ratio": 0.5}),
        ("mp3_compression", {"bitrate_kbps": 16}),
        ("mp3_compression", {"bitrate_kbps": 32}),
        ("echo", {"delay_seconds": 0.3, "decay": 0.3}),
        ("echo", {"delay_seconds": 0.2, "decay": 0.2}),
        ("echo", {"delay_seconds": 0.1, "decay": 0.1}),
    ]

    for name, params in perturbations_to_test:
        try:
            perturbed = perturber.apply_perturbation(audio, name, **params)
            print(f"Applied {name}: shape {perturbed.shape}")
            param_suffix = "_".join([f"{k}_{v}" for k, v in params.items()])

            # Save example (optional)
            output_file = save_dir / f"perturbed_{name}_{param_suffix}.wav"
            sf.write(output_file, perturbed, sr)
            print(f"  Saved to: {output_file}")

        except Exception as e:
            print(f"  Failed to apply {name}: {e}")


if __name__ == "__main__":
    debug_audio = (
        "/home/tst000/projects/LatentGroot/test/dummy_dataset/spk_1246_txt_69_ref_3.wav"
    )
    sample_rate = 16000

    speech_array, sampling_rate = librosa.load(debug_audio, sr=sample_rate)

    aug_fn = RandomAudioAugmentation(sample_rate=sampling_rate)
    speech_array = aug_fn.custom_clip(torch.from_numpy(speech_array)).numpy()
    speech_array = torch.from_numpy(aug_fn(speech_array))
    print(speech_array.shape)

    torchaudio.save(
        "/home/tst000/projects/LatentGroot/test/aug_result.wav",
        speech_array.unsqueeze(0),
        sample_rate,
    )

    # save_dir = Path("/home/tst000/projects/LatentGroot/test/augmentation_test")
    # main(debug_audio, save_dir)
