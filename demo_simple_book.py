from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa, argparse, torch, sys, os
from audioread.exceptions import NoBackendError
import sounddevice as sd
import time

if __name__ == '__main__':

 enc_model_fpath = Path("encoder/saved_models/german_encoder.pt")
 syn_model_fpath = Path("synthesizer/saved_models/german_synthesizer/german_synthesizer.pt")
 voc_model_fpath = Path("vocoder/saved_models/german_vocoder/german_vocoder.pt")

 print("Lade Encoder, Synthesizer und Vocoder ...")
 encoder.load_model(enc_model_fpath)
 synthesizer = Synthesizer(syn_model_fpath)
 vocoder.load_model(voc_model_fpath)

 in_path = './00000561.wav'
 in_text = 'Vater! Der Schläfer ist erwacht!'
 out_path = './gen.wav'

 original_wav, sampling_rate = librosa.load(str(in_path))
 preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

 embed = encoder.embed_utterance(preprocessed_wav)
 print("Erzeuge das Embedding...")

 texts = [in_text]
 embeds = [embed]
 #embeds = [[0] * 256]

 specs = synthesizer.synthesize_spectrograms(texts, embeds)
 spec = specs[0]
 print("Mel Spectrogram erfolgreich erzeugt")

 #generated_wav = vocoder.infer_waveform(spec)
 #generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate),
 #mode="constant")
 #generated_wav = encoder.preprocess_wav(generated_wav)
 generated_wav = Synthesizer.griffin_lim(spec)
 
 print("Starte Audioausgabe.")
 audio_length = librosa.get_duration(generated_wav, sr = 16000)
 sd.play(generated_wav.astype(np.float32), round(synthesizer.sample_rate / 1.0))
 time.sleep(audio_length)
 print("Erledigt.")

 sf.write(out_path, generated_wav.astype(np.float32),
 round(synthesizer.sample_rate / 1.0))

 print("Audiodatei wurde geschrieben.")
