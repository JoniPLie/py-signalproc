# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:40:55 2019

@author: H113864
"""

# Cadzow denoising algorithm 
import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# Generate Clean Signal
N=128
freq=[0.125,0.405]
ak=[15,10]
K=2
x=[0 for i in range(N)] # Initialize input x
for k in range(K):
    x+=np.multiply(ak[k],np.exp(np.multiply([(n) for n in range(128)],complex(0,1)*freq[k]*2*cmath.pi)))
#plt.plot(x.real)
xFFT=np.fft.fft(x)
plt.plot(abs(xFFT))
plt.title('Fourier Transform of Clean Signal')

# Observed Signal
Nobs=512
#xpad=np.append(x,[0 for k in range(Nobs-N)]) # zero pad the clean signal
# the duration where signal is present is irrelevant. Here we introduce delay of 100 samples
xpad=np.append([0 for k in range(100)],np.append(x,[0 for k in range(Nobs-N-100)]))
plt.figure(8)
plt.plot(abs(np.fft.fft(xpad)))
plt.title('Fourier Transform of Zero Padded Clean Signal')
NoiseSig=18
nois = NoiseSig*np.random.randn(1,Nobs)+NoiseSig*np.random.randn(1,Nobs)*1j
y = xpad + np.ravel(nois)
plt.figure(2)
plt.subplot(2,1,1).plot(xpad.real)
plt.ylim(-80,80)
plt.title('Padded Clean Signal (top) & Noise (bottom) - Real Component')
plt.subplot(2,1,2).plot(np.ravel(nois.real))
plt.ylim(-80,80)
plt.figure(3)
plt.plot(abs(np.fft.fft(y)))
plt.title('Fourier Transform of Noisy Signal')

# Cadzow denoising
m = 2*K+1 # column size of Toeplitz matrix
first_col = y[m:Nobs]
first_row = np.flip(y[0:m+1])
YToepMat=toeplitz(first_col,first_row) # form Toeplitz matrix
theta=np.append(first_col,first_row[1:]) # calculate vector theta
T_YToep=np.reshape(YToepMat,(Nobs-m)*(m+1),order='F')
A=np.zeros((len(T_YToep),Nobs))
rowIndex=0
for k in T_YToep: 
    onesIndices=np.where(k==theta) 
    A[rowIndex,onesIndices]=1
    rowIndex+=1
# Iteration
N_iter=20
Y=YToepMat # initialize
for k in range(N_iter):
    u, s, vh = np.linalg.svd(Y, full_matrices=True)
    smat = np.zeros((Nobs-m, m+1), dtype=complex)
    sclean=s
    sclean[K:]=0
    smat[:(m+1), :(m+1)]=np.diag(sclean)
    Y=np.dot(u, np.dot(smat, vh))
    # concatenate
    T_Y=np.reshape(Y,(Nobs-m)*(m+1),order='F')
    theta_hat=np.dot(np.linalg.pinv(A),T_Y)
    y_den = np.append(np.flip(theta_hat[Nobs-m:]),theta_hat[:Nobs-m])
    first_col_den = y_den[m:Nobs]
    first_row_den = np.flip(y_den[0:m+1])
    Y=toeplitz(first_col_den,first_row_den) # form Toeplitz matrix

# Get back the denoised time domain
x_den=np.append(np.flip(Y[0]),Y[1:, 0])
XdenFFT=np.fft.fft(x_den)
plt.figure(4)
plt.plot(abs(XdenFFT))
plt.title('Fourier Transform of Denoised Signal')
