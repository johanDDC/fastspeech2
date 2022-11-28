Fastspeech2
===========
PyTorch implementation of text-to-speech system Fastspeech2. 
Implemented as a home assignment on deep learning in audio course.

Requirements
================

Install requirements by

``pip install -r requirements.txt``

Training
=========

Before training you should do preprocessing (besides you already have preprocessed data).
To do so you should firstly compute statistics by running followig command

``cd preprocessing``

``python3 get_stats.py``

Than carefully record gained statistics in appropriate fields of file `config/fastspeech2.yaml`. 
And than run command

``python3 get_target_pitch_energy.py``

Note, that it can take a significant amount of time.

Also note: steps above are describing a situation, when you already
got dataset and situated it by path `data/LJSpeed-1.1`. If it's not the case,
than run script by path `scripts/download_dataset.sh`.

After all preprocessing stuff is done, you may run training by command

``cd ..``

``python3 train.py``

All the model settings and training process setups are controllable from the file `config/fastspeech2.yaml`.

Inference
==========

Firstly you should set the path to your pretrained model in field
`val.model_path` in file `config/fastspeech2.yaml`. My pretrained checkpoint you may 
get by running `scripts/download_checkpoint.sh`. After that setup your
preferable speed, pitch in energy is neighbour fields. Than set path to
file with text you want to synthesize in a field `val.val_texts`. Than
run

``python3 eval.py``

Generated records will be placed in directory `results`.

