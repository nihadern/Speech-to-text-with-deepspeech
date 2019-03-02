#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import scipy.io.wavfile as wav
from deepspeech import Model
import wave
import subprocess
import shlex
import pyaudio
import os
from queue import Queue
import tello
import time


import numpy as np
try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.50

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 2.10


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


MODELS_PATH = '../models'

model = os.path.join(MODELS_PATH, 'output_graph.rounded.pbmm')
alpha = os.path.join(MODELS_PATH, 'alphabet.txt')
trie = os.path.join(MODELS_PATH, 'trie')
lm = os.path.join(MODELS_PATH, 'lm.binary')



ds = Model(model, N_FEATURES, N_CONTEXT, alpha,BEAM_WIDTH) #model link, cepstrum, context
ds.enableDecoderWithLM(alpha,
                           lm,
                           trie,
                           LM_WEIGHT,
                           VALID_WORD_COUNT_WEIGHT)
print('\nModel ok')
print('\nreading voice')

import pyaudio

fs= 16000

CHUNK = 4096

RECORD_SECONDS = 5
#16000*5 # fixed chunk size



# initialize portaudio


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=CHUNK)

start_time = time.time()

connection = tello.Tello()
connection.send("command")

print(connection.send("battery?"))

try:
	while True:
		# do this as long as you want fresh samples
		numpydata = []
		text = input("Press [ENTER] to continue...")
		print(connection.send("battery?"))
		stream.start_stream()
		for i in range(0, int(fs / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			numpydata.append(np.fromstring(data, dtype=np.int16))
		stream.stop_stream()

		numpydata = np.concatenate(numpydata)
		print(numpydata.shape)

		# ~ if time.time() - start_time > 30:
			# ~ connection.send("land")

		spoken = str(ds.stt(numpydata, fs))
		print(spoken)
		
		def contains(s, x):
			for word in x:
				if word in s:
					return True
			return False
		
		if "land" in spoken:
			print("landing command recieved...")
			connection.send("land")
		elif contains(spoken, ["emergency", "emergenci", "he mergency", "abort", "a board", "aboart", "abandon ship", "kill now", "stop now"]):
			connection.send("emergency")
		elif contains(spoken, ["regina take off", "regina take ouff", "regina take opf"]):
			print("about to send take off command...")
			start_time = time.time()
			connection.send("takeoff")
		elif "regina twist" in spoken:
			print("about to send flip command...")
			connection.send("flip b")
		
		
		
except KeyboardInterrupt:
	# close stream
	stream.stop_stream()
	stream.close()
	p.terminate()
	


# ~ print(len(data))
# ~ print(len(numpydata))

# ~ # close stream
# ~ stream.stop_stream()
# ~ stream.close()
# ~ p.terminate()

# ~ print("Done")



# ~ print('Done')
# ~ #/usr/local/lib/python3.5/dist-packages/deepspeech
# ~ print(len(data))
# ~ print(len(numpydata))

# ~ # close stream
# ~ stream.stop_stream()
# ~ stream.close()
# ~ p.terminate()

# ~ print("Done")



# ~ print('Done')
# ~ #/usr/local/lib/python3.5/dist-packages/deepspeech
