import deepspeech as ds
import numpy as np
import shlex
import subprocess
import sys
model = '/home/nihadern/voice_recog/models/output_graph.pbmm'
alpha = '/home/nihadern/voice_recog/models/alphabet.txt'
trie = '/home/nihadern/voice_recog/models/trie'
lm = '/home/nihadern/voice_recog/models/lm.binary'


LM_WEIGHT = 1.5
VALID_WORD_COUNT_WEIGHT = 2.25
N_FEATURES = 26
N_CONTEXT = 9
BEAM_WIDTH = 512

print('Initializing model...')

model = ds.Model(model, N_FEATURES, N_CONTEXT, alpha, BEAM_WIDTH)
if lm and trie:
    model.enableDecoderWithLM(alpha,
                              lm,
                              trie,
                              LM_WEIGHT,
                              VALID_WORD_COUNT_WEIGHT)
sctx = model.setupStream()

subproc = subprocess.Popen(shlex.split('rec -q -V0 -e signed -L -c 1 -b 16 -r 16k -t raw - gain -2'),
                           stdout=subprocess.PIPE,
                           bufsize=0)
print('You can start speaking now. Press Control-C to stop recording.')

try:
    while True:
        data = subproc.stdout.read(512)
        model.feedAudioContent(sctx, np.frombuffer(data, np.int16))
except KeyboardInterrupt:
    print('Transcription: ', model.finishStream(sctx))
    subproc.terminate()
    subproc.wait()
