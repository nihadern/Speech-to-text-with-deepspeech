import pyaudio
import numpy as np

CHUNKSIZE =  16000*10 # fixed chunk size

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNKSIZE)

# do this as long as you want fresh samples
data = stream.read(CHUNKSIZE)
numpydata = np.fromstring(data, dtype=np.int16)

print(numpydata)

# close stream
stream.stop_stream()
stream.close()
p.terminate()
