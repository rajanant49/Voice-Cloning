# Real-Time Voice Cloning

This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time.

## Setup

### 1. Install Requirements

1. Both Windows and Linux are supported. A GPU is recommended for training and for inference speed, but is not mandatory.
2. Python 3.7 is recommended. Python 3.5 or greater should work, but you'll probably have to tweak the dependencies' versions. I recommend setting up a virtual environment using `venv`, but this is optional.
3. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.
4. Install [PyTorch](https://pytorch.org/get-started/locally/). Pick the latest stable version, your operating system, your package manager (pip by default) and finally pick any of the proposed CUDA versions if you have a GPU, otherwise pick CPU. Run the given command.
5. Install the remaining requirements with `pip install -r requirements.txt`

### 2. Run Wrapper Code

Run wrapper code by :

`python3 wrapper_code.py`

The wrapper code is written on top of the above stated framework to take the input as an audio file, then passing [this](https://1drv.ms/u/s!AssNWNmUSYNtjuAAEIevUoZy8ZnhfA?e=cWum1r) audio input to the code and cloning the voice from the audio file using the framework. The the cloned voice is used to generate audio file for the text using the framework.
