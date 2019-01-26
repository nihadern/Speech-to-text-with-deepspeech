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

import numpy as np
try:
    from shhlex import quote
except ImportError:
    from pipes import quote

LM_WEIGHT = 1.5
VALID_WORD_COUNT_WEIGHT = 2.25
N_FEATURES = 26
N_CONTEXT = 9
BEAM_WIDTH = 512

model = '/home/nihadern/voice_recog/models/output_graph.rounded.pbmm'
micro = '/home/nihadern/voice_recog/male.wav'
alpha = '/home/nihadern/voice_recog/models/alphabet.txt'
trie = '/home/nihadern/voice_recog/models/trie'
lm = '/home/nihadern/voice_recog/models/lm.binary'



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

CHUNKSIZE =  16000*10 # fixed chunk size

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=CHUNKSIZE)

# do this as long as you want fresh samples
data = stream.read(CHUNKSIZE)
numpydata = np.fromstring(data, dtype=np.int16)

print(len(data))
print(len(numpydata))

# close stream
stream.stop_stream()
stream.close()
p.terminate()

print("Done")


print(ds.stt(numpydata, fs))
print('Done')
#/usr/local/lib/python3.5/dist-packages/deepspeech
