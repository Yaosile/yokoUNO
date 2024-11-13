import numpy as np
from numpy import asanyarray as ana
import time

from scipy.io import wavfile
import scipy
import scipy.signal

from PIL import Image

import MelMyBoy
import myOwnLibrary as myJazz
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import feedForwardybackwards as cnn
import os

dira = 'CardSnaps/'

files = [i for i in os.listdir(dira) if i[0] == 'b']

choice = np.random.choice(files)

img = ana(Image.open(dira + choice))[...,::-1]
img = myJazz.rgb2gray(img)

img = Image.fromarray(img.astype(np.uint8))
img.show()


# kernels = []
# kernels.append(ana([
#     [0,0,0],
#     [0,1,0],
#     [0,0,0]
# ]))
# kernels.append(ana([
#     [-1,-1,-1],
#     [-1,8,-1],
#     [-1,-1,-1],
# ]))
# # kernels.append(ana([
# #     [0,-1,0],
# #     [-1,5,-1],
# #     [0,-1,0]
# # ]))
# kernels.append(np.ones((3,3))/9)
# # kernels.append(ana([
# #     [0,0,0],
# #     [0,1,0],
# #     [0,0,0]
# # ]))

# t = ana([
#     [3,10,3],
#     [0,0,0],
#     [-3,-10,-3]
# ])
# kernels.append(t)
# kernels.append(myJazz.rot90(t))
# kernels = ana(kernels).astype(float)
# # for i in range(len(kernels)):
#     # kernels[i] = kernels[i]/np.sum(kernels[i])
# out = []
# og = ana(Image.open('Images/cards.jpg'))
# og = og[:, :og.shape[1]//4].copy().astype(float)
# og = myJazz.scaleImage(og, 512//2,512//2)
# for index in range(len(kernels)):
#     R = og[:,:,0]
#     G = og[:,:,1]
#     B = og[:,:,2]

#     R = myJazz.convolveMultiplication(R,kernels[index])
#     G = myJazz.convolveMultiplication(G,kernels[index])
#     B = myJazz.convolveMultiplication(B,kernels[index])

#     pool = 8

#     R = cnn.getMaxKernel(R,pool)
#     G = cnn.getMaxKernel(G,pool)
#     B = cnn.getMaxKernel(B,pool)

#     card = np.stack((R,G,B), axis=2)
#     # card[card <0] = 0
#     card -= card.min()
#     card /= card.max()
#     card *= 255
#     out.append(card)
#     card = Image.fromarray(card.astype(np.uint8))
#     card.save(f'tout/Convolve{index}.png')

# out = ana(out)
# print(out.shape)
# print(out.reshape(-1).shape)

# # now = time.time_ns()
# # for i in range(100):
# #     myJazz.convolveMultiplication(og, kernel)
# #     # convolve2d(og,kernel)
# #     # scipy.signal.fftconvolve(og, kernel, mode='full')
# # print((time.time_ns() - now)/1e9)


# # # Read the audio file
# # sr, audio = wavfile.read('final.wav')

# # # Normalize audio
# # audio_normalized = audio / np.abs(audio).max()

# # # Set up the figure and grid specification
# # fig = plt.figure(figsize=(15, 8))
# # gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1], height_ratios=[1, 1.5])

# # # Plot the audio waveform
# # ax1 = fig.add_subplot(gs[0, 0])
# # ax1.plot(np.linspace(0, len(audio) / sr, len(audio)), audio_normalized)
# # ax1.set_ylabel('Normalized Amplitude')
# # ax1.set_xlabel('Time (s)')
# # ax1.grid(True)
# # ax1.set_title('Audio Waveform')

# # # Convert to MFCC
# # t = MelMyBoy.convertToMFCC(audio, filterN=100, coeff=80, FFTN=2048)

# # # Plot MFCC response
# # ax2 = fig.add_subplot(gs[1, 0])
# # im = ax2.imshow(t, origin='lower', aspect='auto')
# # ax2.set_ylabel('Coefficient Number')
# # ax2.set_xlabel('Frame')
# # ax2.set_title('MFCC Response')

# # # Add colorbar to a separate axis on the right
# # cbar_ax = fig.add_subplot(gs[:, 1])
# # cbar = fig.colorbar(im, cax=cbar_ax)
# # cbar.set_label('Amplitude')

# # plt.tight_layout(pad=2)
# # plt.show()



# # # fmin = 0
# # # fmax = 44100/2

# # # mel, freq= MelMyBoy.getFilterPoints(fmin, fmax, 20)
# # # filters = MelMyBoy.getTriangleFilters(freq)

# # # plt.figure(figsize=(30,6))
# # # plt.subplot(2,1,1)
# # # plt.ylabel('Amplitude')
# # # plt.plot(np.sum(filters.T, axis=1))
# # # plt.grid(True)
# # # plt.title('Original Filters')

# # # plt.subplot(2,1,2)
# # # plt.ylabel('Amplitude')
# # # plt.xlabel('Bin')
# # # plt.plot(np.sum((filters*(2.0/(mel[2:] - mel[:-2]))[:, None]).T, axis=1))
# # # plt.title('Mel Weighted Filters')
# # # plt.grid(True)
# # # plt.show()








# # # index = 60

# # # sr, audio = wavfile.read('download.wav')
# # # frame = MelMyBoy.audioFrames(audio, sampleRate=sr, windowing=False)[index]
# # # frameWindow = MelMyBoy.audioFrames(audio)[index]

# # # #100
# # # plt.figure(figsize=(30,6))

# # # plt.subplot(2, 2, 1)
# # # plt.plot(frame)
# # # plt.title('Original Frame')
# # # plt.ylabel('Amplitude')
# # # plt.grid(True)

# # # FFT1 = np.abs(myJazz.FFT(frame))

# # # plt.subplot(2, 2, 2)
# # # plt.plot(FFT1)
# # # plt.title('FFT of Original Frame')
# # # plt.grid(True)

# # # plt.subplot(2, 2, 3)
# # # plt.plot(frameWindow)
# # # plt.title('Frame After Windowing')
# # # plt.ylabel('Amplitude')
# # # plt.xlabel('Sample')
# # # plt.grid(True)

# # # FFT2 = np.abs(myJazz.FFT(frameWindow))

# # # plt.subplot(2, 2, 4)
# # # plt.plot(FFT2)
# # # plt.title('FFT of Frame After Windowing')
# # # plt.grid(True)

# # # plt.tight_layout(pad = 6)
# # # plt.xlabel('Bin')
# # # plt.show()