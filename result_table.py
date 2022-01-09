# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import treetable as tt

BASELINES = [
    'WaveUNet',
    'MMDenseLSTM',
    'OpenUnmix',
    'IRM2',
]
EVALS = Path("evals")
LOGS = Path("logs")
BASELINE_EVALS = Path("baselines")
STD_KEY = "seed"

parser = argparse.ArgumentParser("result_table.py")
parser.add_argument("-p",
                    "--paper",
                    action="store_true",
                    help="show results from the paper experiment")
parser.add_argument("-i", "--individual", action="store_true", help="no aggregation by seed")
parser.add_argument("-l", "--latex", action="store_true", help="output easy to copy latex")
parser.add_argument("metric", default="SDR", nargs="?")
args = parser.parse_args()

if args.paper:
    EVALS = Path("results/evals")
    LOGS = Path("results/logs")


def read_track(metric, results, pool=np.nanmedian):
    all_metrics = {}
    for target in results["targets"]:
        source = target["name"]
        metrics = [frame["metrics"][metric] for frame in target["frames"]]
        
        metrics = pool(metrics)
        all_metrics[source] = metrics
    return all_metrics


def read(metric, path, pool=np.nanmedian):
    all_metrics = defaultdict(list)
    for f in path.iterdir():
        if f.name.endswith(".json.gz"):
            results = json.load(gzip.open(f, "r"))
            metrics = read_track(metric, results, pool=pool)
            for source, value in metrics.items():
                all_metrics[source].append(value)
    return {key: np.array(value) for key, value in all_metrics.items()}

top_5_indices = None
def my_read_track(metric, results, song_name):
    all_metrics = {}
    for target in results["targets"]:
        source = target["name"]
        metrics = [frame["metrics"] for frame in target["frames"]]
#         all_metrics[source] = np.nanmedian(metrics)
#         global top_5_indices
#         if top_5_indices is None:
#             top_5_indices = np.argsort(metrics)[-60:-55]
#         print([metric['SDR'] for metric in metrics])
        segment_num = 32
#         segment_num = 20
        all_metrics[source] = {'SDR': metrics[segment_num]['SDR'], 'SAR': metrics[segment_num]['SAR'], 'SIR': metrics[segment_num]['SIR'], 'whole_value_SDR': np.nanmedian([x['SDR'] for x in metrics]), 'whole_value_SAR': np.nanmedian([x['SAR'] for x in metrics]), 'whole_value_SIR': np.nanmedian([x['SIR'] for x in metrics])}
    return all_metrics
printed = False
def my_read(metric, path, pool=np.nanmedian):
    all_metrics = defaultdict(list)
    i = -1
    for f in path.iterdir():
        i += 1
        global printed
        if f.name.endswith(".json.gz") and i == 4:
            if printed == False:
                print(f.name)
                printed = True
            
            results = json.load(gzip.open(f, "r"))
            metrics = my_read_track(metric, results, f.name)
            for source, value in metrics.items():
                all_metrics[source].append(value)
    print(all_metrics['vocals'], end='')

#     max_value = 0
#     max_metric = None
#     for metric in all_metrics['vocals']:
#         if metric['value'] > max_value:
#             max_value = metric['value']
#             max_metric = metric
#     print(str(path).split('/')[1] + '\n' + str(metrics))
    return {key: np.array(value) for key, value in all_metrics.items()}


all_stats = defaultdict(list)
# for name in BASELINES:
#     all_stats[name] = [read(args.metric, BASELINE_EVALS / name / "test")]
for path in EVALS.iterdir():
    results = path / "results" / "test"
    if not results.exists():
        continue
    if not args.paper and not (LOGS / (path.name + ".done")).exists():
        continue
    name = path.name
    model = "Demucs"
    if "tasnet" in name:
        model = "Tasnet"
    if name == "default":
        parts = []
    else:
        parts = [p.split("=") for p in name.split(" ") if "tasnet" not in p]
    if not args.individual:
        parts = [(k, v) for k, v in parts if k != STD_KEY]
    name = model + " " + " ".join(f"{k}={v}" for k, v in parts)
    stats = read(args.metric, results)
    
    
#     model_name_dict = {
#         'pad=True batch_size=4 tasnet=True': 'A1',
#         'pad=True batch_size=2 tasnet=True seq=True': 'A2',
#         'pad=True dwt=True batch_size=4 tasnet=True seq=True': 'A3',
#         'pad=True dwt=True learnable=inverse musdb=musdb18 samples=44100 batch_size=4 tasnet=True seq=True split_valid=True': 'A4',
#         'pad=True dwt=True musdb=musdb18 samples=44100 batch_size=2 tasnet=True seq=True split_valid=True L=10': 'A5',
#         'pad=True dilation_split=True cascade=True batch_size=4 tasnet=True': 'A6',
#         'band_num=2 pad=True batch_size=2 tasnet=True': 'A7',
#         'band_num=2 pad=True dwt=True batch_size=2 tasnet=True': 'A8'
#     }
#     model_name_dict = {
#         'pad=True samples=44100 batch_size=8': 'B1',
#         'band_num=2 pad=True samples=44100 batch_size=4 bottle_dim=128': 'B2',
#         'band_num=2 pad=True samples=44100 batch_size=4': 'B3',
#         'band_num=4 pad=True samples=44100 batch_size=3 remix_group_size=1 bottle_dim=64': 'B4',
#         'band_num=2 pad=True full_band=True samples=44100 batch_size=2 remix_group_size=1 bottle_dim=128': 'B5',
#     }
#     model_name_dict = {
#         'pad=True stronger_enc=True samples=44100 batch_size=6 remix_group_size=3': 'C1',
#         'band_num=2 pad=True stronger_enc=True samples=44100 batch_size=4 remix_group_size=2 bottle_dim=128': 'C2',
#         'band_num=2 pad=True stronger_enc=True concat_features=False samples=44100 batch_size=4 remix_group_size=2 bottle_dim=128': 'C3'
#     }
    model_name_dict = {
        'pad=True gammatone=True transposed_decoder=True samples=44100 batch_size=8 mse=True': 'D1',
        'band_num=2 pad=True gammatone=True transposed_decoder=True samples=44100 batch_size=4 mse=True remix_group_size=2 bottle_dim=128': 'D2',
        'band_num=2 pad=True gammatone=True transposed_decoder=True phase_split_filterbank=True samples=44100 batch_size=4 mse=True remix_group_size=2 bottle_dim=128': 'D3',
        'band_num=2 pad=True gammatone=True transposed_decoder=True post_enc_linear=True samples=44100 batch_size=4 mse=True remix_group_size=2 bottle_dim=128': 'D4'
    }
    print(path.name)
    
    if path.name in model_name_dict.keys():
        l_object = {}
        for metric in ['SIR', 'SAR', 'SDR']:
            stats = read(metric, results)
            l_object[metric] = stats
        np.save(f'{model_name_dict[path.name]}.results', [l_object])
    my_read(args.metric, results)
#     print(f' ~ {name}')
    # if (not stats or len(stats["drums"]) != 50):
    #     print(f"Missing stats for {results}", file=sys.stderr)
    # else:
    all_stats[name].append(stats)

metrics = [tt.leaf("score", ".2f"), tt.leaf("std", ".2f")]
sources = ["drums", "bass", "other", "vocals"]
sources = ["accompaniment", "vocals"]

# mytable = tt.table([tt.leaf("name"), tt.group("all", metrics + [tt.leaf("count")])] +
mytable = tt.table([tt.leaf("name"), tt.group("all", metrics)] +
                   [tt.group(source, metrics) for idx, source in enumerate(sources)])

lines = []
for name, stats in all_stats.items():
    line = {"name": name}
    if 'accompaniment' in stats:
        del stats['accompaniment']
    alls = []
    for source in sources:
        stat = [np.nanmedian(s[source]) for s in stats]
        alls.append(stat)
        
        line[source] = {"score": np.mean(stat), "std": np.std(stat) / len(stat)**0.5}
    alls = np.array(alls)
    line["all"] = {
        "score": alls.mean(),
        "std": alls.mean(0).std() / alls.shape[1]**0.5,
        "count": alls.shape[1]
    }
    lines.append(line)


def latex_number(m):
    out = f"{m['score']:.2f}"
    if m["std"] > 0:
        std = "{:.2f}".format(m["std"])[1:]
        out += f" $\\scriptstyle\\pm {std}$"
    return out


lines.sort(key=lambda x: -x["all"]["score"])
if args.latex:
    for line in lines:
        cols = [
            line['name'],
            latex_number(line["all"]),
            latex_number(line["drums"]),
            latex_number(line["bass"]),
            latex_number(line["other"]),
            latex_number(line["vocals"])
        ]
        print(" & ".join(cols) + r" \\")
else:
    print(tt.treetable(lines, mytable, colors=['33', '0']))
