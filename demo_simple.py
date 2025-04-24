from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from librosa import display
import argparse
import torch
import sys
import os
from audioread.exceptions import NoBackendError
import sounddevice as sd
import time

if __name__ == '__main__':
        
	enc_model_fpath = Path("encoder/saved_models/german_encoder.pt")
	syn_model_fpath = Path("synthesizer/saved_models/german_synthesizer/german_synthesizer.pt")
	voc_model_fpath = Path("vocoder/saved_models/german_vocoder/german_vocoder.pt")
 
	if torch.cuda.is_available():
		device_id = torch.cuda.current_device()
		gpu_properties = torch.cuda.get_device_properties(device_id)
		print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
		"%.1fGb total memory.\n" % 
		(torch.cuda.device_count(),
		device_id,
		gpu_properties.name,
		gpu_properties.major,
		gpu_properties.minor,
		gpu_properties.total_memory / 1e9))
	else:
		print("Using CPU for inference.\n")
    
    
	# Load the models one by one.
	print("Preparing the encoder, the synthesizer and the vocoder...")
	encoder.load_model(enc_model_fpath)
	synthesizer = Synthesizer(syn_model_fpath)
	vocoder.load_model(voc_model_fpath)
	

	in_path = './00000561.wav'
	in_text = 'Vater! Der Schläfer ist erwacht!'
	out_path = './gen.wav'

	original_wav, sampling_rate = librosa.load(str(in_path))
	preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)

	embed = encoder.embed_utterance(preprocessed_wav)
	print("Created the embedding")

	texts = [in_text]
	embeds = [embed]
	#embeds = [[0] * 256]

	specs = synthesizer.synthesize_spectrograms(texts, embeds)
	spec = specs[0]	
	print("Created the mel spectrogram")

	griffin_lim = False
	if not griffin_lim:
		generated_wav = vocoder.infer_waveform(spec)
		generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
		generated_wav = encoder.preprocess_wav(generated_wav)
	else:
		generated_wav = Synthesizer.griffin_lim(spec)
	
	print("Starting audio output")
	audio_length = librosa.get_duration(generated_wav, sr = 14545)
	sd.play(generated_wav.astype(np.float32), round(synthesizer.sample_rate / 1.0))
	time.sleep(audio_length)
	print("Done")
	
	write = True
	if write:
		sf.write(out_path, generated_wav.astype(np.float32), round(synthesizer.sample_rate / 1.0))	
		print("Audio file has been written.")