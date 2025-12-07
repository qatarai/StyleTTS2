import copy
import re

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from monotonic_align import mask_from_lens, maximum_path
from monotonic_align.core import maximum_path_c

# import matplotlib.pyplot as plt
from munch import Munch
from torch import nn


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
        val_list = f.readlines()

    return train_list, val_list


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def log_print(message, logger):
    logger.info(message)
    print(message)


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in "!?.\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv


voice_style_names = [
    "f-qa-1",
    "f-us-1",
    "f-us-2",
    "f-us-3",
    "f-us-4",
    "m-us-1",
    "m-us-2",
    "m-us-3",
    "m-us-4",
]
voice_style_name_to_index = {
    name: index for index, name in enumerate(voice_style_names)
}
