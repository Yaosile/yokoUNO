import numpy as np
from numpy import asanyarray as ana
import feedForwardybackwards as cnn
from scipy.io import wavfile
import pyaudio
import wave
import os

import myOwnLibrary as myJazz

from matplotlib import pyplot as plt

FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1              # Mono channel
RATE = 44100              # 44.1kHz sample rate
CHUNK = 1024              # 1024 samples per frame
RECORD_SECONDS = 1        # Duration to record

WORDS = ['r', 'y', 'g', 'b', 'u']

freq2mel = np.vectorize(lambda x: 2595.0 * np.log10(1.0 + x / 700.0))
mel2freq = np.vectorize(lambda x: 700.0 * (10.0**(x / 2595.0) - 1.0))
def byte2int(x):
    return[int.from_bytes(x[i:i+len(x)//CHUNK], 'little', signed=True) for i in range(0,len(x), len(x)//CHUNK)]

def normaliseAudio(audio):
    audio = audio/np.abs(audio).max()
    return audio

def audioFrames(audio, FFT_n = 2048, hop = 10, sampleRate = 44100):
    window = getWindow(2048)

    audio = np.pad(audio, FFT_n//2,mode='reflect')
    frameLen = (sampleRate * hop + 500) // 1000
    frameTot = (len(audio) - FFT_n) // frameLen + 1
    frames = []
    for n in range(frameTot):
        frames.append(audio[n*frameLen:n*frameLen+FFT_n])

    return ana(frames) * window

def FFTframes(frames):
    FFT = []
    for i in frames:
        FFT.append(np.square(np.abs(myJazz.FFT(i)[:len(i)//2 + 1])))
    return ana(FFT)

def getWindow(FFT_n = 2048):
    window = np.linspace(0,np.pi, FFT_n)
    window = np.sin(window)**2
    return window

def getFilterPoints (fmin, fmax, melFilterNum, FFT_n = 2048, sampleRate = 44100):
    mmin = freq2mel(fmin)
    mmax = freq2mel(fmax)
    mamas = np.linspace(mmin, mmax, melFilterNum + 2)
    freqs = mel2freq(mamas)
    return freqs, ((FFT_n + 1) / sampleRate * freqs).astype(int)

def getTriangleFilters(freqs, FFT_n = 2048):
    filters = []
    for n in range(1,len(freqs)-1):
        start,peak,end = freqs[n-1], freqs[n], freqs[n+1]
        q1 = np.linspace(0,0,start-0)
        q2 = np.linspace(0,1,peak-start)
        q3 = np.linspace(1,0,end-peak)
        q4 = np.linspace(0,0,FFT_n//2 + 1 - end)
        temp = np.hstack((q1,q2,q3,q4))
        filters.append(temp)
    return ana(filters)

def dct(num, len):
    b = np.empty((num,len))
    b[0,:] = 1.0/(len**0.5)

    s = np.arange(1,2*len,2)*np.pi/(2.0*len)
    for i in range(1,num):
        b[i,:] = np.cos(i*s)*((2.0/len)**0.5)
    return b

def convertToMFCC(audio, hop=15, sampleRate = 44100, filterN = 10, coeff=40):
    audio = normaliseAudio(audio)
    frames = audioFrames(audio, hop = hop)
    FFT = FFTframes(frames)
    melFreqs,freqs=getFilterPoints(0, sampleRate/2, filterN)
    filters=getTriangleFilters(freqs)*(2.0/(melFreqs[2:] - melFreqs[:-2]))[:, None]
    filteredOutput = (np.dot(filters, FFT.T))
    dctcoefficients = dct(coeff, filterN)
    final = np.dot(dctcoefficients, filteredOutput)
    return final[:coeff//2,:]

def trainCNN(layersN):
    files = os.listdir('AudioRecordings')
    data = []
    correct = []
    for f in files:
        data.append(np.load(f'AudioRecordings/{f}'))
        t = [0 for i in range(len(WORDS))]
        t[WORDS.index(f[0])] = 1
        correct.append(t)

    audioSample = data[0]
    weights = []
    for i in range(layersN-1):
        weights.append(np.load(f'audioWeights/{i}.npy'))

    # weights = cnn.generateWeights(len(audioSample), len(audioSample), len(WORDS), layers=layersN)

    L2, weights = cnn.backProp(data, correct, weights, 0.01, 500, 0.01, 1)
    for i, weight in enumerate(weights):
        np.save(f'audioWeights/{i}.npy', weight)
    np.save('error.npy', L2)

def getAudioRecording():
    print('starting capture')
    audio = pyaudio.PyAudio()
    OUTPUT_FILENAME = "AudioRecordings/output.wav"
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    recording = []
    print('GO')
    while True:
        data = stream.read(CHUNK)
        data = byte2int(data)
        if np.max(data) > 2000:
            recording += data
            break
    print('STARTING')
    for _ in range(1,int(RATE/CHUNK*RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data = byte2int(data)
        recording += data
    print('END')
    recording = ana(recording)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    final = convertToMFCC(recording)
    return final.reshape(-1)

def doOnline():
    weights = []
    for i in range(3):
        weights.append(np.load(f'audioWeights/{i}.npy'))
    files = os.listdir('AudioRecordings')
    data = []
    correct = []
    for f in files:
        data.append(np.load(f'AudioRecordings/{f}'))
        t = [0 for i in range(len(WORDS))]
        t[WORDS.index(f[0])] = 1
        correct.append(t)
    while True:
        print()
        now = getAudioRecording()
        guess = np.round(cnn.feedForward([now], weights), 4)
        print(guess)
        print(f'Guessed: {WORDS[np.argmax(guess)]}')
        word = input('please enter the correct word: ').lower()[0]
        if word not in WORDS:
            break
        
        t = [0 for i in range(len(WORDS))]
        t[WORDS.index(word)] = 1
        correct.append(t)

        n = len(os.listdir('AudioRecordings'))
        np.save(f'AudioRecordings/{word}{n}.npy', now)
        data.append(np.load(f'AudioRecordings/{word}{n}.npy'))

def classify():
    weights = []
    for i in range(3):
        weights.append(np.load(f'audioWeights/{i}.npy'))
    while True:
        print()
        now = getAudioRecording()
        guess = np.round(cnn.feedForward([now], weights), 4)
        print(guess)
        print(f'Guessed: {WORDS[np.argmax(guess)]}')

def checkHealth():
    weights = []
    for i in range(3):
        weights.append(np.load(f'audioWeights/{i}.npy'))
    # weights = cnn.generateWeights(1340, 1340, 5, 4)
    files = os.listdir('AudioRecordings')
    data = []
    correct = []
    order = []
    for f in files:
        data.append(np.load(f'AudioRecordings/{f}'))
        t = [0 for i in range(len(WORDS))]
        t[WORDS.index(f[0])] = 1
        correct.append(t)
        order.append(f[0])
    
    output = cnn.feedForward(data, weights)
    tally = {i:0 for i in WORDS}
    tallycorrect = {i:0 for i in WORDS}
    for i,j in zip(output, correct):
        if np.argmax(i) == np.argmax(j):
            tallycorrect[WORDS[np.argmax(j)]] += 1
        tally[WORDS[np.argmax(j)]] += 1
    
    for i in WORDS:
        print(i, tallycorrect[i],tally[i])


if __name__ == '__main__':
    # data = []
    # for i in WORDS:
    #     sr, audio = wavfile.read(f'AudioRecordings/{i}.wav')
    #     final = convertToMFCC(audio, sampleRate=sr)
    #     data.append(final.reshape(-1))
    # trainCNN(data, 5)
    # test = getAudioRecording()
    # np.save('AudioRecordings/r0.npy', test)
    # doOnline()
    # trainCNN(4)
    # classify()
    # green = np.load('AudioRecordings/g48.npy').reshape(20,-1)
    # blue = np.load('AudioRecordings/b33.npy').reshape(20,-1)
    # plt.imshow(blue)
    # plt.show()

    # classify()
    # doOnline()
    trainCNN(4)
    error = np.load('error.npy')
    plt.plot(error)
    plt.show()
