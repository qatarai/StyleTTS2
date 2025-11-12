import os

import nltk
from cached_path import cached_path

nltk.download("punkt")
nltk.download("punkt_tab")

import torch
from scipy.io.wavfile import write

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random

random.seed(0)

import numpy as np

np.random.seed(0)

# load packages
import random
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from models import *
from munch import Munch
from nltk.tokenize import word_tokenize
from text_utils import TextCleaner
from torch import nn
from utils import *

textclenaer = TextCleaner()


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s.squeeze(1), ref_p.squeeze(1)], dim=1)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    # print("MPS would be available but cannot be used rn")
    pass
    # device = 'mps'

import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)
# phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))


# config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
config = yaml.safe_load(
    open(str(cached_path("hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/config.yml")))
)

current_dir = os.path.dirname(os.path.abspath(__file__))

# load pretrained ASR model
ASR_config = config.get("ASR_config", False)
ASR_config = os.path.join(current_dir, ASR_config)
ASR_path = config.get("ASR_path", False)
ASR_path = os.path.join(current_dir, ASR_path)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get("F0_path", False)
F0_path = os.path.join(current_dir, F0_path)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert

BERT_path = config.get("PLBERT_dir", False)
BERT_path = os.path.join(current_dir, BERT_path)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config["model_params"])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
params_whole = torch.load(
    str(
        cached_path(
            "hf://yl4579/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
        )
    ),
    map_location="cpu",
)
params = params_whole["net"]

for key in model:
    if key in params:
        print("%s loaded" % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict

            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(
        sigma_min=0.0001, sigma_max=3.0, rho=9.0
    ),  # empirical parameters
    clamp=False,
)

voice_styles = {
    name: compute_style(current_dir + f"/voices/{name}.wav")
    for name in voice_style_names
}


def inference(
    text,
    s_prev=None,
    s_ref=None,
    alpha=0.3,
    beta=0.7,
    t=0.7,
    diffusion_steps=5,
    embedding_scale=1,
    noise=None,
):
    if isinstance(text, str):
        text = [text]

    for i, txt in enumerate(text):
        txt = txt.strip()
        txt = txt.replace('"', "")
        txt = txt.replace("``", '"')
        txt = txt.replace("''", '"')
        text[i] = txt

    batch_size = len(text)

    ps = global_phonemizer.phonemize(text)
    ps = (word_tokenize(p) for p in ps)
    ps = (" ".join(p) for p in ps)

    tokens = (textclenaer(p) for p in ps)
    tokens = (t.insert(0, 0) or t for t in tokens)

    with torch.no_grad():
        tokens = [torch.LongTensor(t).to(device) for t in tokens]
        input_lengths = torch.LongTensor([len(t) for t in tokens]).to(device)
        max_length = torch.max(input_lengths).item()
        tokens = [
            torch.nn.functional.pad(t, (0, max_length - len(t)), value=0)
            for t in tokens
        ]
        tokens = torch.stack(tokens).to(device)

        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        if s_ref is None:
            s_ref = voice_styles[voice_style_names[0]]

        if s_ref.shape[0] == 1 and batch_size > 1:
            s_ref = s_ref.expand(batch_size, -1)
        elif s_ref.shape[0] != batch_size:
            raise ValueError(
                f"s_ref batch size ({s_ref.shape[0]}) must match text batch size ({batch_size})"
            )

        s_pred = sampler(
            noise
            if noise is not None
            else torch.randn((batch_size, 256)).to(device).unsqueeze(1),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps,
            features=s_ref,
        )

        if s_prev is not None:
            if s_prev.shape[0] == 1 and batch_size > 1:
                s_prev = s_prev.expand(batch_size, -1)
            elif s_prev.shape[0] != batch_size:
                raise ValueError(
                    f"s_prev batch size ({s_prev.shape[0]}) must match text batch size ({batch_size})"
                )

            # convex combination of previous and predicted style
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, :, 128:].squeeze(1)
        ref = s_pred[:, :, :128].squeeze(1)

        if s_ref is not None:
            if s_ref.shape[0] == 1 and batch_size > 1:
                s_ref = s_ref.expand(batch_size, -1)
            elif s_ref.shape[0] != batch_size:
                raise ValueError(
                    f"s_ref batch size ({s_ref.shape[0]}) must match text batch size ({batch_size})"
                )

            # convex combination of reference and predicted style
            s = alpha * s + (1 - alpha) * s_ref[:, 128:]
            ref = beta * ref + (1 - beta) * s_ref[:, :128]

        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze(-1)).clamp(min=1)

        max_phonemes = pred_dur.shape[1]

        # Create alignment matrices for each batch item
        pred_aln_trg_list = []
        frame_lengths = []
        for b in range(batch_size):
            seq_length = int(input_lengths[b].item())
            total_frames = int(pred_dur[b].sum().item())
            frame_lengths.append(total_frames)

            pred_aln_trg_seq = torch.zeros(seq_length, total_frames, device=device)
            c_frame = 0
            for i in range(seq_length):
                frame_dur = int(pred_dur[b, i].item())
                pred_aln_trg_seq[i, c_frame : c_frame + frame_dur] = 1
                c_frame += frame_dur
            pred_aln_trg_list.append(pred_aln_trg_seq)

        # Pad alignment matrices to max frame length
        max_frames = max(frame_lengths)
        pred_aln_trg = torch.zeros(batch_size, max_phonemes, max_frames, device=device)
        for b in range(batch_size):
            seq_length = int(input_lengths[b].item())
            pred_aln_trg[b, :seq_length, : frame_lengths[b]] = pred_aln_trg_list[b]

        # encode prosody
        en = d.transpose(-1, -2) @ pred_aln_trg.to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref)

        # Return list of outputs, one per batch item
        # Calculate output length proportionally: F0/N conv downsamples by 2, then generator upsamples
        # Since upsampling factor is constant, we can use proportional slicing
        outputs = []
        for b in range(batch_size):
            # Calculate output length proportionally based on frame_lengths
            # F0/N conv downsamples by 2, so effective frames = frame_lengths[b] / 2
            # Then upsample by constant factor, so output length is proportional
            actual_length = int(frame_lengths[b] * out.shape[-1] / max_frames)
            outputs.append(
                out[b, :actual_length].cpu().numpy()[..., :-100].astype(np.float32)
            )

        s_pred = s_pred.cpu().numpy().astype(np.float32)
        s_pred = [s_pred[b] for b in range(batch_size)]

    return outputs, s_pred
