#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:13:02 2020

@author: datascience
"""

import cv2
import numpy as np
import os
import time
import hdf5storage


def resize_image(path, size=250):
    '''Opens and reduces the image to a squared defined size. '''
    img = cv2.imread(path)
    #height, width, depth = np.shape(img)
    img = cv2.resize(img,(size,size), interpolation=cv2.INTER_CUBIC)
    return img

def img2_binvector(img, size=250):
    '''Read the image, convert each pixel to its binary representation and 
    transforms the image in a binary vector. '''
    bin_vals = np.zeros(shape=((size**2)*3, 8), dtype='uint8')
    #bin_vals_lst = list()
    i=0
    for v3 in img.swapaxes(2,0).swapaxes(1,2):
        for v2 in np.nditer(v3.T.copy(order='C')):
            #if (i%250)==0:
            #    print(i, end=' ')
            bins = np.binary_repr(v2, width=8)
            #bin_vals_lst.append(bins)
            bin_vals[i,:] = np.array(list(map(int, bins)))
            i+=1
    bin_vector = bin_vals.flatten(order='C')
    #np.savetxt("bins.csv", bin_vector, delimiter=",")
    return bin_vector

def project_Xmat(X_mat, proj_file):
    ''' Projects the Orthonormal Matrix in the images matrix to reduce dimensionality. '''
    orthMat = hdf5storage.loadmat(proj_file)
    idx_name = list(orthMat)[0]
    orthMat = orthMat[idx_name].astype('float')
    X_proj  = np.dot(X_mat,orthMat)
    return X_proj

def apply_transforms(img_path, size=250):
    img = resize_image(img_path, size)
    img = img2_binvector(img, size)
    return img
            
def from_folder(path, proj_path, size=250, y_val=[1,0,2]):
    '''Reads the parent folder to get the internal folders, process the images
    and generates the X and y numpy objects. '''
    paths = sorted([f.path for f in os.scandir(path) if f.is_dir()])
    lens_paths = [len([f.path for f in os.scandir(g) if f.is_file()]) for g in paths]
    X_mat = np.zeros(shape=(sum(lens_paths), (size**2)*8*3), dtype='uint8')
    y_mat = np.repeat(np.array(y_val), lens_paths)
    i=0
    for folder in paths:
        print("--- Starting - folder: {0} ---".format(os.path.basename(folder)))
        start_time = time.time()
        imgs = [f.path for f in os.scandir(folder) if f.is_file()]
        for img_path in imgs:
            X_mat[i,:] = apply_transforms(img_path, size)
            i+=1
        tot_in_sec = time.time() - start_time
        print("--- {0} seconds - folder: {1} ---".format(str(tot_in_sec), os.path.basename(folder)))
    X_mat = project_Xmat(X_mat, proj_path)
    return X_mat, y_mat

def from_image(path, proj_path, size=250, y_val=[1,0,2]):
    


if not('filepath' in locals() or 'projpath' in locals()):
    filepath = input("Origin path: ")
    projpath = input("Orthonormal Projection Matrix full path (folder+name+extension): ")
X, y = set_X_y(filepath, projpath, size=10)
#np.savez_compressed('tmp/Xy50', X=X, y=y)
#loaded = np.load('tmp/Xy1.npz')
#loaded.files
#X=loaded['arr_0']
#y=loaded['arr_1']