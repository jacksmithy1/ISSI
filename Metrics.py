#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:06:51 2023

@author: lilli
"""

import numpy as np

def Vec_corr_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    num = np.sum(np.multiply(B, b))
    div = np.sqrt(np.sum(np.multiply(B, B))*np.sum(np.multiply(b, b)))
    
    return num/div


def Cau_Schw_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    N = np.size(B)
    num = np.multiply(B, b)
    div = np.reciprocal(np.multiply(abs(B), abs(b)))
    temp = np.sum(np.multiply(num, div))
    
    return temp/N


def Norm_vec_err_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    num = np.sum(abs(np.subtract(B, b)))
    div = np.sum(np.abs(B))
    
    return num/div
    
    
def Mean_vec_err_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    N = np.size(B)
    num = abs(np.subtract(B, b))
    div = abs(np.reciprocal(B))
    temp = np.sum(np.multiply(num, div))
    
    return temp/N


def Mag_ener_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    Bx = B[:,:,:,1][0,0]
    By = B[:,:,:,0][0,0]
    Bz = B[:,:,:,2][0,0]
    bx = b[:,:,:,1][0,0]
    by = b[:,:,:,0][0,0]
    bz = b[:,:,:,2][0,0]
    
    num = np.sqrt(np.dot(bx, bx)+np.dot(by, by)+np.dot(bz, bz))
    div = np.sqrt(np.dot(Bx, Bx)+np.dot(By, By)+np.dot(Bz, Bz))
    
    return num/div


# def main():
    
#     B_ref = np.zeros((2,2,2,3))
#     B_rec = np.zeros((2,2,2,3))
    
#     eps = 0.00001
    
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 for m in range(3):
#                     B_ref[i,j,k,m] = np.cos(i)*np.sin(j+np.pi/2.0)*(k+1)*(i+1)
#                     B_rec[i,j,k,m] = B_ref[i,j,k,m]+(-1)**i*eps
            
#     #print('B_ref', B_ref)
#     #print('B_rec', B_rec)
    
#     print('B_ref max', B_ref.max())
#     print('B_ref min', B_ref.min())
    
#     Vecmet = Vec_corr_metric(B_ref, B_rec)
#     print(Vecmet)
#     CSmet = Cau_Schw_metric(B_ref, B_rec)
#     print(CSmet)
#     NVmet = Norm_vec_err_metric(B_ref, B_rec)
#     print(NVmet)
#     MVmet = Mean_vec_err_metric(B_ref, B_rec)
#     print(MVmet)
#     MEmet = Mag_ener_metric(B_ref, B_rec)
#     print(MEmet)