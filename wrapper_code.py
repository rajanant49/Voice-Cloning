import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder


def wrapper(audio_file):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    ensure_default_models(Path("saved_models"))
    encoder.load_model(Path('saved_models/default/encoder.pt'))
    synthesizer = Synthesizer(Path('saved_models/default/synthesizer.pt'))
    vocoder.load_model(Path('saved_models/default/vocoder.pt'))
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    mels = synthesizer.synthesize_spectrograms(texts, embeds)
    mel = np.concatenate(mels, axis=1)
    no_action = lambda *args: None
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    num_generated = 0
    try:
        
        preprocessed_wav = encoder.preprocess_wav(audio_file)        

        embed = encoder.embed_utterance(preprocessed_wav)
        text = "Hi, welcome to Interactly, a no coding interactive video creation platform to create the personalized video experiences."
        texts = [text]
        embeds = [embed]

        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]

        generated_wav = vocoder.infer_waveform(spec)

        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        generated_wav = encoder.preprocess_wav(generated_wav)

        filename = "generated_Voice.wav"
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)


    except Exception as e:
        pass

if __name__=='__main__':
    audio_file = Path(input("Enter Audio File path:\n").replace("\"", "").replace("\'", ""))
    wrapper(audio_file)