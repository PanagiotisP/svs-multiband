# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("convtasnet", description="Train and evaluate Conv TasNet.")
    parser.add_argument(
        "--raw",
        type=Path,
        help="Path to raw audio, can be faster, see python3 -m demucs.raw to extract.",
        default=Path("../../../../../gpu-data/ppap/raw_musdb"))
    parser.add_argument("--no_raw", action="store_const", const=None, dest="raw")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--band_num", type=int, default=1)
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--dilation_split", action="store_true")
    parser.add_argument("--cascade", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--full_band", action="store_true")
    parser.add_argument("--sinc_conv", action="store_true")
    parser.add_argument("--gammatone", action="store_true")
    parser.add_argument("--transposed_decoder", action="store_true")
    parser.add_argument("--post_enc_linear", action="store_true")
    parser.add_argument("--phase_split_filterbank", action="store_true")
    parser.add_argument("--frozen_unsupervised_enc_dec", action="store_true")
    parser.add_argument("--frozen_waveunet_enc_dec", action="store_true")
    parser.add_argument("--frozen_original_enc_dec", action="store_true")
    parser.add_argument(
        "--pretrained",
        choices=['M1', 'M2', 'M3', 'M4', 'M5', 'S1', 'S2'],
        help="Pretrained model for evaluation",
    )
    parser.add_argument(
        "--dwt_latent",
        action="store_true",
        help="Adds a DWT layer after the encoder(s) to be applied on the latent representation",
    )
    parser.add_argument(
        "--dwt",
        action="store_true",
        help="Adds a DWT layer before the encoder(s) to be applied on the natural signal",
    )
    parser.add_argument(
        "--learnable",
        type=str,
        default="none",
        help="Adds a learnable layer to match DWT transformation",
    )
    parser.add_argument(
        "--stronger_enc",
        action="store_true",
        help="Use Samuel's stronger encoder",
    )
    parser.add_argument(
        "--no_concat_features",
        dest="concat_features",
        action="store_false",
        help="Use Samuel's stronger encoder",
    )
    parser.add_argument("-m", "--musdb", type=Path, help="Path to musdb root",
        default=Path("../../../../../gpu-data/ppap/musdb18"))
    parser.add_argument("--metadata", type=Path, default=Path("metadata/musdb.json"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--audio_channels", type=int, default=2)
    parser.add_argument("--samples",
                        default=22050 * 4,
                        type=int,
                        help="number of samples to feed in")
    parser.add_argument(
        "--data_stride",
        default=44100,
        type=int,
        help="Stride for chunks, shorter = longer epochs",
    )
    parser.add_argument("-w", "--workers", default=4, type=int, help="Loader workers")
    parser.add_argument("--eval_workers", default=0, type=int, help="Final evaluation workers")
    parser.add_argument(
        "-d",
        "--device",
        help="Device to train on, default is cuda if available else cpu",
    )
    parser.add_argument("--eval_cpu", action="store_true", help="Eval on test will be run on cpu.")
    parser.add_argument("--dummy", help="Dummy parameter, useful to create a new checkpoint file")
    parser.add_argument(
        "--test",
        help="Just run the test pipeline + one validation. "
        "This should be a filename relative to the models/ folder.",
    )

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")

    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints"),
        help="Folder where to store checkpoints etc",
    )
    parser.add_argument(
        "--pretrained_dict",
        type=Path,
        default=Path("pretrained_dict"),
        help="Folder where to store checkpoints etc",
    )
    parser.add_argument(
        "--evals",
        type=Path,
        default=Path("evals"),
        help="Folder where to store evals and waveforms",
    )
    parser.add_argument("--save",
                        action="store_true",
                        help="Save estimated for the test set waveforms")
    parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs"),
                        help="Folder where to store logs")
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help="Folder where to store trained models",
    )
    parser.add_argument(
        "-R",
        "--restart",
        action="store_true",
        help="Restart training, ignoring previous run",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="Repeat the train set, longer epochs",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mse", action="store_true", help="Use MSE instead of L1")
    parser.add_argument("--sisnr", action="store_true", help="Use sisnr instead of L1")
    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        default=True,
        help="No data augmentation",
    )
    parser.add_argument(
        "--remix_group_size",
        type=int,
        default=4,
        help="Shuffle sources using group of this size. Useful to somewhat "
        "replicate multi-gpu training "
        "on less GPUs.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="Number of random shifts used for random equivariant stabilization.",
    )
    parser.add_argument(
        "--no_glu",
        action="store_false",
        default=True,
        dest="glu",
        help="Replace all GLUs by ReLUs",
    )
    # Tasnet options
    parser.add_argument("--tasnet", action="store_true")

    parser.add_argument("--seq", action="store_true")
    parser.add_argument(
        "--no_split_valid",
        action="store_false",
        default=True,
        dest="split_valid",
        help="Predict chunks by chunks for valid and test. Required for tasnet",
    )
    parser.add_argument("--stacks", type=int, default=8)
    parser.add_argument("--racks", type=int, default=3)
    parser.add_argument("--enc_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--bottle_dim", type=int, default=256)
    parser.add_argument("--instr_num", type=int, default=2)
    parser.add_argument("--transform_window", type=int, default=20)

    parser.add_argument("--show",
                        action="store_true",
                        help="Show model architecture, size and exit")
    parser.add_argument("--save_model", action="store_true")

    return parser


def get_name(parser, args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    ignore_args = set([
        "checkpoints",
        "deterministic",
        "epochs",
        "eval",
        "evals",
        "eval_cpu",
        "eval_workers",
        "logs",
        "master",
        "rank",
        "restart",
        "save",
        "save_model",
        "show",
        "valid",
        "workers",
        "world_size",
        "raw",
    ])
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in ignore_args:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}={value.name}")
            else:
                parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    return name
