# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import sys
from concurrent import futures

import musdb
import museval
import torch as th
import tqdm
from scipy.io import wavfile
from torch import distributed

from .raw import MusDBSet
import numpy as np

from .utils import apply_model


def evaluate(
    model,
    musdb_path,
    eval_folder,
    workers=2,
    device="cpu",
    rank=0,
    save=False,
    shifts=0,
    split=False,
    check=True,
    world_size=1,
    samplerate=22050,
    channels=2,
    sisnr=False
):
    """
    Evaluate model using museval. Run the model
    on a single GPU, the bottleneck being the call to museval.
    """

    source_names = ["drums", "bass", "other", "vocals"]
    source_names = ["accompaniment", "vocals"]
    output_dir = eval_folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = eval_folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # we load tracks from the original musdb set
    test_set = MusDBSet(
        musdb.DB(musdb_path, subsets=["test"], is_wav=False),
        channels=channels,
        samplerate=samplerate,
    )

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    pendings = []
    with futures.ProcessPoolExecutor(workers or 1) as pool:
        for index in tqdm.tqdm(range(rank, len(test_set), world_size), file=sys.stdout):
            name, streams = test_set[index]
            mix = streams[0]
            vocals = streams[4]
            accompaniment = streams[1] + streams[2] + streams[3]

            out = json_folder / f"{name}.json.gz"
            if out.exists():
                continue

            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()

            estimates = apply_model(model,
                                    mix.to(device),
                                    shifts=shifts,
                                    split=split)
            
            if sisnr:
                a_l = estimates[:, 0].cpu().numpy().T
                a_r = estimates[:, 1].cpu().numpy().T

                b_l = mix[0].cpu().numpy()
                b_r = mix[1].cpu().numpy()

                
                
                sol_l = np.linalg.lstsq(a_l, b_l, rcond=None)[0]
                sol_r = np.linalg.lstsq(a_r, b_r, rcond=None)[0]
                
                e_l = a_l * sol_l
                e_r = a_r * sol_r

                estimates = th.stack([th.from_numpy(e_l.T), th.from_numpy(e_r.T)], dim=1)  # shape: (instrument, channel, time)

            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.transpose(1, 2)
            references = th.stack([accompaniment.t(), vocals.t()])
            references = references.numpy()
            estimates = estimates.cpu().numpy()
            if save:
                folder = eval_folder / "wav/test" / name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(source_names, estimates):
                    wavfile.write(str(folder / (name + ".wav")), 44100, estimate)

            if workers:
                pendings.append((name, pool.submit(museval.evaluate, references, estimates)))
            else:
                pendings.append((name, museval.evaluate(references, estimates)))
            del references, mix, estimates, streams

        for track_name, pending in tqdm.tqdm(pendings, file=sys.stdout):
            if workers:
                pending = pending.result()
            sdr, isr, sir, sar = pending
            track_store = museval.TrackStore(win=44100, hop=44100, track_name=track_name)
            for idx, target in enumerate(source_names):
                values = {
                    "SDR": sdr[idx].tolist(),
                    "SIR": sir[idx].tolist(),
                    "ISR": isr[idx].tolist(),
                    "SAR": sar[idx].tolist(),
                }

                track_store.add_target(target_name=target, values=values)
                json_path = json_folder / f"{track_name}.json.gz"
                gzip.open(json_path, "w").write(track_store.json.encode("utf-8"))
    if world_size > 1:
        distributed.barrier()
