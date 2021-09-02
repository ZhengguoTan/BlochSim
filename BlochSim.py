#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:55:42 2019
based on Brian Hargreaves Bloch Simulator in MATLAB
http://mrsrl.stanford.edu/~brian/bloch/
@author: zhengguo.tan@gmail.com
"""

import numpy
from numpy import cos, sin, exp, pi

def zrot(phi):
    return numpy.array([[cos(phi), -sin(phi), 0], 
                        [sin(phi),  cos(phi), 0], 
                        [0       , 0        , 1]])
    
def yrot(phi):
    return numpy.array([[ cos(phi), 0, sin(phi)],
                        [0        , 1, 0       ],
                        [-sin(phi), 0, cos(phi)]])
    
def xrot(phi):
    return numpy.array([[1, 0       , 0        ],
                        [0, cos(phi), -sin(phi)], 
                        [0, sin(phi),  cos(phi)]])
    
def throt(phi, theta):
    return zrot(theta).dot(xrot(phi).dot(zrot(-theta)))
    
def freeprecess(T, T1, T2, df):
    """ 
    Function simulates free precession and decay
    over a time interval T, given relaxation times T1 and T2
    and off-resonance df.  Times in ms, off-resonance in Hz.
    """
    phi = 2*pi*df*T/1000;
    E1 = exp(-T/T1);
    E2 = exp(-T/T2);
    
    Afp = numpy.array([[E2, 0 , 0],
                       [0 , E2, 0], 
                       [0 , 0 , E1]])
    Afp = numpy.matmul(Afp, zrot(phi))
    
    Bfp = numpy.array([[0], [0], [1-E1]])
    #Bfp = numpy.array([[0, 0, 1-E1]])
    
    return Afp, Bfp

def sssignal(FA, T1, T2, TE, TR, df):
    """
    Calculate the steady state signal at TE for repeated
    excitations given T1,T2,TR,TE in ms.  dfreq is the resonant
    frequency in Hz.  flip is in radians.
    Given
        M1 - the magnetization before the tip
        M2 - the magnetization after the tip
        M3 - the magnetization at TE
    Then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Atr * M3 + Btr
    Solve for M3 -- steady state signal intensity
        M1 = M2
    """
    
    Rflip = yrot(FA)
    Atr,Btr = freeprecess(TR-TE, T1, T2, df)
    Ate,Bte = freeprecess(TE   , T1, T2, df)
    
    Mss = numpy.dot(numpy.linalg.inv(numpy.eye(3) 
                    - numpy.dot(Ate,Rflip).dot(Atr)),
                    numpy.dot(Ate,Rflip).dot(Btr) + Bte)
    
    Msig = Mss[0,0] + 1j*Mss[1,0]
    
    return Msig, Mss

def sesignal(T1, T2, TE, TR, df):
    """
    Calculate the steady state signal at TE for spin echo
    sequence (90x - 180y - 90x - 180y ...) given T1,T2,TR,TE in ms.  
    dfreq is the resonant frequency in Hz.  flip is in radians.
    Given
        M1 - the magnetization before the tip
        M2 - the magnetization after the tip
        M3 - the magnetization at TE
    Then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Atr * M3 + Btr
    Solve for M3 -- steady state signal intensity
        M1 = M2
    """
    
    Rflip = yrot(pi/2)
    Rrefo = xrot(pi)
    
    Atr ,Btr  = freeprecess(TR-TE, T1, T2, df)
    Ateh,Bteh = freeprecess(TE/2 , T1, T2, df)
    
    fwd = Ateh.dot(Rrefo.dot(Ateh.dot(Rflip)))
    
    lhs = numpy.eye(3) - numpy.dot(fwd,Atr)
    rhs = numpy.dot(fwd,Btr) + Ateh.dot(Rrefo.dot(Bteh)) + Bteh
    
    Mss = numpy.dot(numpy.linalg.inv(lhs), rhs)
    
    Msig = Mss[0,0] + 1j*Mss[1,0]
    
    return Msig, Mss

def fsesignal(T1, T2, TE, TR, df, ETL):
    
    Rflip = yrot(pi/2)
    Rrefo = xrot(pi)
    
    Ateh,Bteh = freeprecess(TE/2     , T1, T2, df)
    Atr ,Btr  = freeprecess(TR-ETL*TE, T1, T2, df)
    
    # Since ETL varies, let's keep a "running" A and B.  We'll
    # calculate the steady-state signal just after the tip, Rflip.
    
    # initilization
    A = numpy.eye(3)
    B = numpy.zeros((3,1))
    
    # for each echo, we propagate A and B by looking at 
    # the interval "TE/2 --- refoc --- TE/2"
    for n in range(0,ETL):
        A = Ateh.dot(Rrefo.dot(Ateh.dot(A)))
        B = Bteh + Ateh.dot(Rrefo.dot(Ateh.dot(B) + Bteh))
    
    # propagate A and B to just after flip
    A = Rflip.dot(Atr.dot(A))
    B = Rflip.dot(Atr.dot(B) + Btr)
    
    M = numpy.dot(numpy.linalg.inv(numpy.eye(3) - A), B)
    
    Mss = numpy.zeros((3,ETL))
    for n in range(0,ETL):
        M = Ateh.dot(Rrefo.dot(Ateh.dot(M)+Bteh)) + Bteh
        Mss[:,[n]] = M
    
    return Mss

def gssignal(FA, T1, T2, TE, TR, df, phi):
    """
    Calculate the steady state signal at TE for gradient-spoiled
    sequence given T1,T2,TR,TE in ms.  dfreq is the resonant
    frequency in Hz.  flip is in radians.  phi is the angle by 
    which the magnetization is dephased at the end of the TR
    Given
        M1 - the magnetization before the tip
        M2 - the magnetization after the tip
        M3 - the magnetization at TE
    Then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Rdeph * (Atr * M3 + Btr)
    Solve for M3 -- steady state signal intensity
        M1 = M2
    """
    
    Rflip = yrot(FA)
    Rdeph = zrot(phi)
    Atr,Btr = freeprecess(TR-TE, T1, T2, df)
    Ate,Bte = freeprecess(TE   , T1, T2, df)
    
    tmp = Ate.dot(Rflip.dot(Rdeph))
    lhs = numpy.eye(3) - tmp.dot(Atr)
    rhs = Bte + tmp.dot(Btr)
    
    Mss = numpy.dot(numpy.linalg.inv(lhs), rhs)
    
    Msig = Mss[0,0] + 1j*Mss[1,0]
    
    return Msig, Mss

def gresignal(FA, T1, T2, TE, TR, df):
    
    N = 100
    M = numpy.zeros((3,N))
    
    phi = ((numpy.arange(N)+1)/N - 0.5) * 4.*pi
    
    for n in range(0,N):
        Msig,Mss = gssignal(FA, T1, T2, TE, TR, df, phi[n])
        M[:,[n]] = Mss
        
    Mss = numpy.mean(M, axis=1)
    Msig = Mss[0] + 1j*Mss[1]
    
    return Msig, Mss
    
def ssfp(FA, T1, T2, TE, TR, df, phi):
    """
    Calculate the steady state signal at TE for redocused-ssfp
    sequence given T1,T2,TR,TE in ms.  dfreq is the resonant
    frequency in Hz.  flip is in radians.  phi is the angle by 
    which the magnetization is dephased at the end of the TR
    Given
        M1 - the magnetization before the tip
        M2 - the magnetization after the tip
        M3 - the magnetization at TE
    Then
        M2 = Rflip * M1
        M3 = Ate * M2 + Bte
        M1 = Rdeph * (Atr * M3 + Btr)
    Solve for M3 -- steady state signal intensity
        M1 = M2
    """
    
    Rflip = yrot(FA)
    Rdeph = zrot(phi)
    Atr,Btr = freeprecess(TR-TE, T1, T2, df)
    Ate,Bte = freeprecess(TE   , T1, T2, df)
    
    tmp = Ate.dot(Rflip.dot(Rdeph))
    lhs = numpy.eye(3) - tmp.dot(Atr)
    rhs = Bte + tmp.dot(Btr)
    
    Mss = numpy.dot(numpy.linalg.inv(lhs), rhs)
    
    Msig = Mss[0,0] + 1j*Mss[1,0]
    
    return Msig, Mss

def spgrsignal(FA, T1, T2, TE, TR, df, Nex, inc):
    
    Nspin = 100
    phi = ((numpy.arange(Nspin)+1)/Nspin - 0.5) * 2.*numpy.pi

    Mspin = numpy.zeros((3,Nspin))
    Mspin[2,:] = 1.

    Atr,Btr = freeprecess(TR-TE, T1, T2, df)
    Ate,Bte = freeprecess(TE   , T1, T2, df)

    P     = 0
    P_inc = inc

    Mte = numpy.zeros((Nex,1), dtype=complex)

    for n in range(0,Nex):
        Rth = throt(FA, P)
        Mspin = Ate.dot(Rth.dot(Mspin)) + Bte
    
        Mte[n] = numpy.mean(Mspin[0,:] + 1j*Mspin[1,:]) * numpy.exp(-1j*P)
    
        Mspin = Atr.dot(Mspin) + Btr
    
        for m in range(0,Nspin):
            Mspin[:,m] = zrot(phi[m]).dot(Mspin[:,m])
    
        P = P + P_inc
        P_inc = P_inc + inc
        
    return Mte[Nex-1]

# %%





