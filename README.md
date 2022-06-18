# MULTI-BAND MASKING FOR WAVEFORM-BASED SINGING VOICE SEPARATION

Support material for the paper "MULTI-BAND MASKING FOR WAVEFORM-BASED SINGING VOICE SEPARATION" accepted in EUSIPCO 2022.
The code and repo structure was forked [from this repo](https://github.com/facebookresearch/demucs).

## Paper Abstract

Singing Voice Separation (SVS) is the task of isolating the vocal
component from a given musical mixture. In recent years there
has been an increase in both the quantity and quality of Singing
Voice Separation techniques that operate in the waveform domain
and have an encoder-separator-decoder structure, where the separa-
tor processes a latent representation of the waveform produced by
the encoder. In this work, we propose a parallel multi-band modi-
fication for this family of architectures, that splits the latent repre-
sentation provided by the encoder in multiple sub-bands and then
processes each band in isolation, using multiple separators, so as
to discover and better exploit the internal correlations of each sub-
band. We investigate the effect of our proposed modification on
Conv-TasNet, a widely used architecture adhering to the encoder-
separator-decoder paradigm. The results indicate that the proposed
modification improves the overall performance, without increasing
the network size significantly, and offer insights on its scaling capa-
bilities as well as its applicability in other architectures that follow
this general paradigm.

## Recreation of results

To recreate each one of the trained models, the following commands need to be executed.

* M1

``python run.py --world_size 1 --pad --samples 44100 --batch_size 8``
* M2

``python run.py --world_size 1 --pad --samples 44100 --batch_size 4 --band_num 2 --bottle_dim 128``
* M3

``python run.py --world_size 1 --pad --samples 44100 --batch_size 2 --remix_group_size 1 --bottle_dim 128 --full_band --band_num 2``
* M4

``python run.py --world_size 1 --pad --samples 44100 --batch_size 3 --remix_group_size 1 --bottle_dim 64 --band_num 4``
* M5

``python run.py --world_size 1 --pad --samples 44100 --band_num 2 --batch_size 4 --remix_group_size 2 --bottle_dim 128 --frozen_original_enc_dec``
* S1

``python run.py --world_size 1 --pad --samples 44100 --stronger_enc --batch_size 6 --remix_group_size 3``
* S2

``python run.py --world_size 1 --pad --samples 44100 --band_num 2 --stronger_enc --batch_size 4 --remix_group_size 2 --bottle_dim 128``
