#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import scipy.io.wavfile as wav
from deepspeech import Model
import wave
import subprocess
import shlex
import numpy as np
try:
    from shhlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)



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

fin = wave.open(micro, 'rb')
fs = fin.getframerate()
if fs != 16000:
    print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
    fs, audio = convert_samplerate(micro)
else:
    audio = naudiop.frombuffer(fin.readframes(fin.getnframes()), np.int16)

audio_length = fin.getnframes() * (1/16000)
fin.close()


print(ds.stt(audio, fs))
print('Done')
#/usr/local/lib/python3.5/dist-packages/deepspeech
