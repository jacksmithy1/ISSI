#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:10:41 2023

@author: lilli
"""

import numpy as np
from datetime import datetime  
import time  
import scipy

def fft_coeff_Seehafer(b_back, k2_arr, nresol_x, nresol_y, nf_max):
    
    anm_Seehafer = 0.0*k2_arr
    
    signal1 = np.fft.fft2(b_back)/nresol_x/nresol_y
    signal = np.fft.fftshift(signal1)
    
    #print(signal[int(nresol/2.0):nresol,int(nresol/2.0):nresol].size)
    
    #print('fft with seehafer', signal[int(nresol/2.0):nresol,int(nresol/2.0):nresol])
    
    #time_stamp = time.time()
    #date_time = datetime.fromtimestamp(time_stamp)
    #str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    #print("Current time FFT DONE", str_date_time)
    
    for ix in range(0,nresol_x,2):
        for iy in range(1,nresol_y,2):
            temp = signal[iy,ix]
            signal[iy,ix] = -temp
            
    for ix in range(1,nresol_x,2):
        for iy in range(0,nresol_y,2):
            temp = signal[iy,ix]
            signal[iy,ix] = -temp
            
    Mx = int(nresol_x/2)
    My = int(nresol_y/2)
    
    for ix in range(1,nf_max):
        for iy in range(1,nf_max):
            anm_Seehafer[iy,ix] = (-signal[My+iy,Mx+ix]+signal[My+iy,Mx-ix]+signal[My-iy,Mx+ix]-signal[My-iy,Mx-ix]).real/k2_arr[iy,ix]
            #anm_Seehafer[ix,iy] = (-signal[Mx+ix,My+iy]+signal[Mx+ix,My-iy]+signal[Mx-ix,My+iy]-signal[Mx-ix,My-iy]).real/k2_arr[ix,iy]
        
    #time_stamp = time.time()
    #date_time = datetime.fromtimestamp(time_stamp)
    #str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    #print("Current time COEFF DONE", str_date_time)
        
    return anm_Seehafer, signal