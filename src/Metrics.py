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
    div = np.sqrt(np.sum(np.multiply(B, B)) * np.sum(np.multiply(b, b)))

    return num / div


def Cau_Schw_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    N = np.size(B)
    num = np.multiply(B, b)
    div = np.reciprocal(np.multiply(abs(B), abs(b)))
    temp = np.sum(np.multiply(num, div))

    return temp / N


def Norm_vec_err_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    num = np.sum(abs(np.subtract(B, b)))
    div = np.sum(np.abs(B))

    return num / div


def Mean_vec_err_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    N = np.size(B)
    num = abs(np.subtract(B, b))
    div = abs(np.reciprocal(B))
    temp = np.sum(np.multiply(num, div))

    return temp / N


def Mag_ener_metric(B, b):
    """
    B : B_ref
    b : B_rec
    """
    Bx = B[:, :, :, 1][0, 0]
    By = B[:, :, :, 0][0, 0]
    Bz = B[:, :, :, 2][0, 0]
    bx = b[:, :, :, 1][0, 0]
    by = b[:, :, :, 0][0, 0]
    bz = b[:, :, :, 2][0, 0]

    num = np.sqrt(np.dot(bx, bx) + np.dot(by, by) + np.dot(bz, bz))
    div = np.sqrt(np.dot(Bx, Bx) + np.dot(By, By) + np.dot(Bz, Bz))

    return num / div
