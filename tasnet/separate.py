# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import argparse
import hashlib
import sys
from pathlib import Path

import requests
import torch as th
import numpy as np

import tqdm
from scipy.io import wavfile

from .audio import AudioFile
from .utils import apply_model, load_model


def download_file(url, target):
    """
    Download a file with a progress bar.

    Arguments:
        url (str): url to download
        target (Path): target path to write to
        sha256 (str or None): expected sha256 hexdigest of the file
    """
    def _download():
        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length', 0))

        with tqdm.tqdm(total=total_length, ncols=120, unit="B", unit_scale=True) as bar:
            with open(target, "wb") as output:
                for data in response.iter_content(chunk_size=4096):
                    output.write(data)
                    bar.update(len(data))

    try:
        _download()
    except:  # noqa, re-raising
        if target.exists():
            target.unlink()
        raise


def verify_file(target, sha256):
    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)
    signature = hasher.hexdigest()
    if signature != sha256:
        print(
            f"Invalid sha256 signature for the file {target}. Expected {sha256} but got "
            f"{signature}.\nIf you have recently updated the repo, it is possible "
            "the checkpoints have been updated. It is also possible that a previous "
            f"download did not run to completion.\nPlease delete the file '{target.absolute()}' "
            "and try again.",
            file=sys.stderr)
        sys.exit(1)


def encode_mp3(wav, path, bitrate=320, samplerate=44100, channels=2, verbose=False):
    try:
        import lameenc
    except ImportError:
        print(
            "Failed to call lame encoder. Maybe it is not installed? "
            "On windows, run `python.exe -m pip install -U lameenc`, "
            "on OSX/Linux, run `python3 -m pip install -U lameenc`, "
            "then try again.",
            file=sys.stderr)
        sys.exit(1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    mp3_data = encoder.encode(wav.tostring())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def main():
    parser = argparse.ArgumentParser("tasnet.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    parser.add_argument("-n",
                        "--name",
                        help="Model name. See README.md for the list of pretrained models. ")
                        
    parser.add_argument("-Q",
                        "--quantized",
                        action="store_true",
                        dest="quantized",
                        default=False,
                        help="Load the quantized model rather than the quantized version. "
                        "Quantized model is about 4 times smaller but might worsen "
                        "slightly quality.")
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--sisnr", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("models"),
                        help="Path to trained models. "
                        "Also used to store downloaded pretrained models")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=0,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--nosplit",
                        action="store_false",
                        default=True,
                        dest="split",
                        help="Apply the model to the entire input at once rather than "
                        "first splitting it in chunks of 10 seconds. Will OOM with Tasnet "
                        "but will work fine for Demucs if you have at least 16GB of RAM.")
    parser.add_argument("--float32",
                        action="store_true",
                        help="Convert the output wavefile to use pcm f32 format instead of s16. "
                        "This should not make a difference if you just plan on listening to the "
                        "audio but might be needed to compute exactly metrics like SDR etc.")
    parser.add_argument("--int16",
                        action="store_false",
                        dest="float32",
                        help="Opposite of --float32, here for compatibility.")
    parser.add_argument("--mp3", action="store_true", help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate", default=320, type=int, help="Bitrate of converted mp3.")
    parser.add_argument("--multi",
                        default=False,
                        action="store_true",
                        help="Multi instrument separation")

    args = parser.parse_args()
    name = args.name + ".th"
    if args.quantized:
        name += ".gz"

    model_path = args.models / name
    if not model_path.is_file():
        print(f"No pretrained model {args.name}", file=sys.stderr)
        sys.exit(1)
    model = load_model(model_path).to(args.device)
    if args.quantized:
        args.name += "_quantized"
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    source_names = ["drums", "bass", "other", "vocals"
                    ] if args.multi else ["accompaniment", "vocals"]
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        print(f"Separating track {track}")
        wav = AudioFile(track).read(streams=0,
                                    samplerate=model.samplerate,
                                    channels=2).to(args.device)
        # Round to nearest short integer for compatibility with how MusDB load audio with stempeg.
        wav = (wav * 2**15).round() / 2**15
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model,
                              wav,
                              shifts=args.shifts,
                              split=args.split,
                              progress=True)
        
        if args.sisnr:
            a_l = sources[:, 0].cpu().numpy().T
            a_r = sources[:, 1].cpu().numpy().T

            b_l = wav[0].cpu().numpy()
            b_r = wav[1].cpu().numpy()                

            sol_l = np.linalg.lstsq(a_l, b_l, rcond=None)[0]
            sol_r = np.linalg.lstsq(a_r, b_r, rcond=None)[0]

            e_l = a_l * sol_l
            e_r = a_r * sol_r

            sources = th.stack([th.from_numpy(e_l.T), th.from_numpy(e_r.T)], dim=1).cuda()  # shape: (instrument, channel, time)

        
        
        sources = sources * ref.std() + ref.mean()

        track_folder = out / track.name.split(".")[0]
        track_folder.mkdir(exist_ok=True)
        for source, name in zip(sources, source_names):
            if args.mp3 or not args.float32:
                source = (source * 2**15).clamp_(-2**15, 2**15 - 1).short()
            source = source.cpu().transpose(0, 1).numpy()
            stem = str(track_folder / name)
            if args.mp3:
                encode_mp3(source,
                           stem + ".mp3",
                           bitrate=args.mp3_bitrate,
                           samplerate=model.samplerate,
                           channels=model.audio_channels,
                           verbose=args.verbose)
            else:
                wavname = str(track_folder / f"{name}.wav")
                wavfile.write(wavname, model.samplerate, source)


if __name__ == "__main__":
    main()
