#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:29:21 2019

@author: ztan
"""

import sys
sys.path.append('/media/radon_nfs4_home/reco/sim')

import matplotlib.pyplot as plt
import BlochSim
import numpy
import random

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

numpy.set_printoptions(precision=4)

# %%
"""
A-4. Rotations: Precession and Excitation
"""

# %%
Rth = BlochSim.throt(numpy.pi/4, numpy.pi/6)
print(Rth)

# %%
"""
A-5. Free Precession Simulation
"""

# %% Excercise A-5
dT = 1
T = 1000
N = (numpy.ceil(T/dT) + 1).astype(int)
df = 10
T1 = 600
T2 = 100

[A,B] = BlochSim.freeprecess(dT, T1, T2, df)

M = numpy.zeros((3,N))
M[0,0] = 1

for k in range(1,N):
    M[:,[k]] = numpy.dot(A, M[:,[k-1]]) + B

time = numpy.arange(N) * dT
plt.plot(time,M[0,:],'b-', time,M[1,:],'r--', time,M[2,:],'g-.')
plt.legend(['M_x', 'M_y', 'M_z'])
plt.show()

# %%
"""
B-1. Saturation Recovery
"""

# %% Excercise B-1a
T1 = 600
T2 = 100
TR = 500
TE = 1

M = numpy.array([[0],[0],[1]])
M = numpy.dot(BlochSim.yrot(numpy.pi/3), M)

A,B = BlochSim.freeprecess(TE,T1,T2,0)
M = numpy.dot(A, M) + B

# %% Excercise B-1b
M = numpy.array([[0],[0],[1]])

# 1st excitation
M = numpy.dot(BlochSim.yrot(numpy.pi/3), M)
# free precession
A,B = BlochSim.freeprecess(TR,T1,T2,0)
M = numpy.dot(A, M) + B

# 2nd excitation
M = numpy.dot(BlochSim.yrot(numpy.pi/3), M)
# free precession
A,B = BlochSim.freeprecess(TE,T1,T2,0)
M = numpy.dot(A, M) + B

# %% Excercise B-1c
dT = 1
NEx = 10
NTr = round(TR/dT)
N = (numpy.ceil(NEx * TR) + 1).astype(int)

# free precession
A,B = BlochSim.freeprecess(dT,T1,T2,0)

M = numpy.zeros((3,N))
M[2,0] = 1

cnt = 0;
for n in range(0,NEx):
    # excitation
    M[:,[cnt]] = numpy.dot(BlochSim.yrot(numpy.pi/3), M[:,[cnt]])
    
    for k in range(0,NTr):
        # free precession
        cnt = cnt + 1
        M[:,[cnt]] = numpy.dot(A, M[:,[cnt-1]]) + B
        
time = numpy.arange(N) * dT
plt.plot(time,M[0,:],'b-', time,M[1,:],'r--', time,M[2,:],'g-.')
plt.show()

# %% Excercise B-1d

M = numpy.array([[0],[0],[1]])

Rflip = BlochSim.yrot(numpy.pi/3)
A,B = BlochSim.freeprecess(TR,T1,T2,0)

M = numpy.dot(numpy.linalg.inv(numpy.eye(3) - numpy.dot(A, Rflip)), B)

# %% Excercise B-1e
Msig,Mss = BlochSim.sssignal(numpy.pi/3,T1,T2,TE,TR,0)


# %%
"""
B-2. Spin-Echo Sequences
"""

# %% Excercise B-2a
T1 = 600
T2 = 100
df = 10
TR = 500
TE = 50
dT = 1

N1 = round(TE/2/dT)
N2 = round((TR-TE/2)/dT)

M = numpy.zeros((3,N1+N2))
M[2,0] = 1 # initial magnetization

Rflip1 = BlochSim.yrot(numpy.pi/2)
Rflip2 = BlochSim.xrot(numpy.pi)
A,B = BlochSim.freeprecess(dT, T1, T2, df)

M[:,[1]] = numpy.dot(A, numpy.dot(Rflip1, M[:,[0]])) + B

for k in range(2,N1+1):
    M[:,[k]] = numpy.dot(A, M[:,[k-1]]) + B

M[:,[1+N1]] = numpy.dot(A, numpy.dot(Rflip2, M[:,[N1]])) + B

for k in range(2,N2):
    M[:,[k+N1]] = numpy.dot(A, M[:,[k-1+N1]]) + B

time = numpy.arange(N1+N2) * dT
axi = plt.subplot(111)
axi.plot(time,M[0,:],'b-' , label='$M_x$')
axi.plot(time,M[1,:],'r--', label='$M_y$')
axi.plot(time,M[2,:],'g-.', label='$M_z$')
axi.legend(loc='upper right')
plt.show()

# %% Excercise B-2b
T1 = 600
T2 = 100
TR = 500
TE = 50
dT = 1

N1 = round(TE/2/dT)
N2 = round((TR-TE/2)/dT)

Nspin = 10

sig = numpy.zeros((Nspin,N1+N2),dtype=complex)

Rflip1 = BlochSim.yrot(numpy.pi/2)
Rflip2 = BlochSim.xrot(numpy.pi)

df_max = 50
df_min = -50
# df = numpy.zeros((Nspin,1))

time = numpy.arange(N1+N2) * dT

random.seed(1)
for s in range(0, Nspin):
    # initial magnetization
    M1 = numpy.zeros((3,N1+N2))
    M1[2,0] = 1

    # off-resonance
    df = df_min + random.random() * (df_max - df_min)
    print('%1d-th spin with the off-resonance %.3f Hz' %(s,df))
    
    # free precession
    A,B = BlochSim.freeprecess(dT, T1, T2, df)
    
    M1[:,[1]] = numpy.dot(A, numpy.dot(Rflip1, M1[:,[0]])) + B
    
    for k in range(2,N1+1):
        M1[:,[k]] = numpy.dot(A, M1[:,[k-1]]) + B
        
    M1[:,[1+N1]] = numpy.dot(A, numpy.dot(Rflip2, M1[:,[N1]])) + B
    
    for k in range(2,N2):
        M1[:,[k+N1]] = numpy.dot(A, M1[:,[k-1+N1]]) + B
        
    sig[[s],:] = M1[[0],:] + 1j*M1[[1],:]
    
    ax1 = plt.subplot(211)
    ax1.plot(time,numpy.absolute(sig[s,:]),'-', label='%1d magnitude' %s)
    
    ax2 = plt.subplot(212)
    ax2.plot(time,numpy.angle(sig[s,:]),'-', label='%1d phase' %s)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

axi = plt.subplot(111)
axi.plot(time,numpy.absolute(numpy.mean(sig,axis=0)),'g-', 
         label='net magnitude')
axi.legend(loc='upper right')
plt.show()

# %% Excercise B-2c
T1 = 600
T2 = 100
TE = 50
TR = 1000

Msig,Mss = BlochSim.sesignal(T1, T2, TE, TR, 0)

print('$M_x$ = %.4f' %Mss[0,0])

# %% Excercise B-2d
T1 = 600
T2 = 100
df = 0
TE = 50
TR = 1000
ETL = 8

Mss_eco = BlochSim.fsesignal(T1, T2, TE, TR, df, ETL)

Mag_eco = numpy.zeros((1,ETL), dtype=complex)

print('fse echo magnitude: ')
for n in range(0,ETL):
    Mag_eco[0,n] = Mss_eco[0,n] + 1j*Mss_eco[1,n]
    print('%.4f ' %(numpy.absolute(Mag_eco[0,n])))


# %%
"""
B-3. Gradient-Spoiled Sequences
"""

# %% Excercise B-3a
FA = numpy.pi/3
T1 = 600
T2 = 100
TE = 2
TR = 10
df = 0
phi = numpy.pi/2

Msig,Mss = BlochSim.gssignal(FA,T1,T2,TE,TR,df,phi)

print('gradient-spoiled steady-state signal: ')
print(Mss)

# %% Excercise B-3b
FA = numpy.pi/3
T1 = 600
T2 = 100
TE = 2
TR = 10
df = 0

Msig,Mss = BlochSim.gresignal(FA, T1, T2, TE, TR, df)

print(Mss)

# %%
"""
B-4. Steady-Sate Free-Precession
"""

# %% Excercise B-4a
T1 = 600
T2 = 100
FA = numpy.pi/3
TR = 10

# generate df from -100 to 100 Hz
df = numpy.linspace(-100,100,num=201)

TE = numpy.array([0., 2.5, 5., 7.5, 10.])

sig_ssfp = numpy.zeros((TE.size, df.size), dtype=complex)

for m in range(0, TE.size):
    for n in range(0, df.size):
        Msig,Mss = BlochSim.sssignal(FA, T1, T2, TE[m], TR, df[n])
        sig_ssfp[m,n] = Msig
    
    ax1 = plt.subplot(211)
    ax1.plot(df,numpy.absolute(sig_ssfp[m,:]),'-', label='TE = %2d ms' %TE[m])
    
    ax2 = plt.subplot(212)
    ax2.plot(df,numpy.angle(sig_ssfp[m,:]),'-', label='TE = %2d ms' %TE[m])

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

# %% Excercise B-4b
T1 = 600
T2 = 100
FA = numpy.pi/3
TR = 10

# generate df from -100 to 100 Hz
df = numpy.linspace(-100,100,num=201)

TE = numpy.array([0., 2.5, 5., 7.5, 10.])

sig_gre  = numpy.zeros((TE.size, df.size), dtype=complex)

for m in range(0, TE.size):
    for n in range(0, df.size):
        Msig,Mss = BlochSim.gresignal(FA, T1, T2, TE[m], TR, df[n])
        sig_gre[m,n] = Msig
    
    ax1 = plt.subplot(211)
    ax1.plot(df,numpy.absolute(sig_gre[m,:]),'-', label='TE = %4.1f ms' %TE[m])
    plt.ylim([0, 0.15])
    
    ax2 = plt.subplot(212)
    ax2.plot(df,numpy.angle(sig_gre[m,:]),'-', label='TE = %4.1f ms' %TE[m])
    plt.ylim([-numpy.pi, numpy.pi])

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

# %% Excercise B-4c
T1 = 600
T2 = 100
FA = numpy.pi/3

TR = numpy.array([2,6,10])
df = numpy.linspace(-500,500,num=1001)

sig_ssfp = numpy.zeros((TR.size, df.size), dtype=complex)
for m in range(0, TR.size):
    for n in range(0, df.size):
        Msig,Mss = BlochSim.sssignal(FA, T1, T2, TR[m]/2., TR[m], df[n])
        sig_ssfp[m,n] = Msig
    
    ax1 = plt.subplot(211)
    ax1.plot(df,numpy.absolute(sig_ssfp[m,:]),'-', label='TR = %.1f ms' %TR[m])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    ax2 = plt.subplot(212)
    ax2.plot(df,numpy.angle(sig_ssfp[m,:]),'-', label='TR = %.1f ms' %TR[m])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

# %% Excercise B-4d
T1 = 600
T2 = 100
FA = numpy.pi/3
TR = 5.0
TE = 2.5

phi = numpy.array([0, numpy.pi/2, numpy.pi, numpy.pi*1.5])
df  = numpy.linspace(-500,500,num=1001)

sig_ssfp = numpy.zeros((phi.size, df.size), dtype=complex)
for m in range(0, phi.size):
    for n in range(0, df.size):
        Msig,Mss = BlochSim.ssfp(FA, T1, T2, TE, TR, df[n], phi[m])
        sig_ssfp[m,n] = Msig
    
    ax1 = plt.subplot(211)
    ax1.plot(df,numpy.absolute(sig_ssfp[m,:]),'-', label='phi = %.2f' %phi[m])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    ax2 = plt.subplot(212)
    ax2.plot(df,numpy.angle(sig_ssfp[m,:]),'-', label='phi = %.2f' %phi[m])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

# %%
"""
B-5. RF-Spoiled Sequences
"""

# %% Excercise B-5a
Nspin = 100
phi = ((numpy.arange(Nspin)+1)/Nspin - 0.5) * 2.*numpy.pi

Mspin = numpy.zeros((3,Nspin))
Mspin[2,:] = 1.

T1 = 600
T2 = 100
TR = 10
TE = 2
df = 0

Atr,Btr = BlochSim.freeprecess(TR-TE, T1, T2, df)
Ate,Bte = BlochSim.freeprecess(TE   , T1, T2, df)

FA    = numpy.pi/6
inc   = numpy.pi*117/180
P     = 0
P_inc = inc

Nex = 100
Aex = numpy.arange(Nex)

Mte = numpy.zeros((Nex,1), dtype=complex)

for n in range(0,Nex):
    Rth = BlochSim.throt(FA, P)
    Mspin = Ate.dot(Rth.dot(Mspin)) + Bte
    
    Mte[n] = numpy.mean(Mspin[0,:] + 1j*Mspin[1,:]) * numpy.exp(-1j*P)
    
    Mspin = Atr.dot(Mspin) + Btr
    
    for m in range(0,Nspin):
        Mspin[:,m] = BlochSim.zrot(phi[m]).dot(Mspin[:,m])
    
    P = P + P_inc
    P_inc = P_inc + inc

ax1 = plt.subplot(211)
ax1.plot(Aex, numpy.absolute(Mte), '-')
plt.xlabel('Excitations')
plt.ylabel('Magnitude')
    
ax2 = plt.subplot(212)
ax2.plot(Aex, numpy.angle(Mte), '-')
plt.xlabel('Excitations')
plt.ylabel('Phase')

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.show()

# %% Excercise B-5b
T1 = 600
T2 = 100
TR = 10
TE = 2
df = 0

Nex = 100
inc = numpy.pi*117/180

FA = numpy.linspace(0,90,num=91)
M  = numpy.zeros((FA.size,1), dtype=complex)

for n in range(0,FA.size):
    M[n] = BlochSim.spgrsignal(FA[n]*numpy.pi/180, T1, T2, TE, TR, df, Nex, inc)
    
ax1 = plt.subplot(111)
ax1.plot(FA, numpy.absolute(M), '-')
plt.xlabel('Flip Angle (deg)')
plt.ylabel('Signal Magnitude')

plt.show()

# %%