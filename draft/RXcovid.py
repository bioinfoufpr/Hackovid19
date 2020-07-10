#!/usr/bin/env python3
from sklearn import preprocessing
import numpy as np
import cv2
import os
import time
import Augmentor
#import glob
import struct
import hdf5storage

laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

contour = np.array((
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]), dtype="int")

# construct the kernel bank, filter2D function
kernelBank = (
        ("laplacian", laplacian),
        ("contour", contour)
)


class TransfParam:
    def __init__(self, nsample=2,szimg=250,pool='maxpool',filtertype='none',szslidwin=3,curr_datetime='None'):
        self.nsample = nsample
        self.szimg = szimg
        self.pool = pool
        self.filtertype = filtertype
        self.szslidwin = szslidwin
        self.curr_datetime = curr_datetime


def normalize(inp):
    min_max_scaler = preprocessing.MinMaxScaler()
    out = min_max_scaler.fit_transform(inp)
    return out


def resize_image(path, size=250):
    '''Opens and reduces the image to a squared defined size. '''
    img = cv2.imread(path)
    img = cv2.resize(img,(size,size), interpolation=cv2.INTER_CUBIC)
    return img


def img2_binvector(img, size=250):
    '''Read the image, convert each pixel to its binary representation and 
    transforms the image in a binary vector. '''
    bin_vals = np.zeros(shape=((size**2)*3, 8), dtype='uint8')
    i=0
    for v3 in img.swapaxes(2,0).swapaxes(1,2):
        for v2 in np.nditer(v3.T.copy(order='C')):
            bins = np.binary_repr(v2, width=8)
            bin_vals[i,:] = np.array(list(map(int, bins)))
            i+=1
    bin_vector = bin_vals.flatten(order='C')
    return bin_vector


def img2_bin(vec):
    '''Read the vectorized image and convert each value in vector to its binary 
    representation (a binary vector). '''
    outvec = np.zeros(shape=(len(vec)*32), dtype='uint8')
    i=0
    for k in vec:
         aux = format(struct.unpack('!I', struct.pack('!f', k))[0], '032b')
         outvec[i:i+32] = [int(i) for i in str(aux)]
         i=i+32
    return outvec  

#filepath = '/home/datascience/Documents/IA_Wonders/hackcovid19/X-Ray Image DataSet/'
#X, y = set_X_y(filepath)
#np.savez_compressed('tmp/Xy50', X=X, y=y)
#loaded = np.load('tmp/Xy50.npz')


def apply_mask(image, kerneltype):   
    ''' Receive an image and apply a 3x3 mask/filter.
    The options are contour, laplacian and none. '''
    if kerneltype=='contour':
        kernel = contour
    elif kerneltype=='laplacian':
        kernel = laplacian
    elif kerneltype=='none':
        return image
    else: 
        print('kernel type unrecognized')
        return image
        
    # catch the dimensions of the image
    (iH, iW) = image.shape[:2] # lines(1) and columns(2)
    (kH, kW) = kernel.shape[:2]
    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32") # create zeros matrix with image size


    # SLIDIND WINDOW
    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            # AQUI PODE SER O SWeeP
            k = (roi * kernel).sum()
            # store the apply_maskd value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    return output


def augment(origin,output,nsample):
    '''receive a folder and a number of desired samples and create a set of 
    'augmented' images using some standard parameters. The Augmentor package 
    comes from an article published in the journal Bioinfromatics.'''
    # Augmentation pagkagepath
    # bibliography in: Bloice,etal,2019_Augmentor.pdf
    # and Augmentor_man.pdf
    #https://github.com/mdbloice/Augmentor
    #import Augmentor
    # Initializang augmentation
    p = Augmentor.Pipeline(origin)
    p.ground_truth(output) # save the generated images here
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
    p.sample(nsample) # number of augmented images based on specifications
    p.process() # start generation
#    string = "mv "+origin+"/output/* "+origin+"/  ;  rm -r "+origin+"/output"
#    os.system(string)


def slidding(image,typetransf,szwin):
    '''Slidding window - szwin = size of window, ex. szwin=3, than 3x3 window'''
    # catch the dimensions of the image
    (nL, nC) = image.shape[:2] # lines(1) and columns(2)
    output = np.zeros(nC*nL, dtype="float32") # create zeros matrix with image size
    
    if typetransf=='maxpool': # usar aqui as funções em linha?
        func_transf = lambda x: x.max()
    elif typetransf=='averagepool':
        func_transf = lambda x: x.mean()
    else: 
        print('transformation type unrecognized')
        return image
    
    # SLIDDING WINDOW
    k=0
    for x in range(0,(nL-szwin)):
        for y in range(0,(nC-szwin)):
            minimatrix = image[x:x+szwin,y:y+szwin]
            output[k] = func_transf(minimatrix)
            k=k+1

    return output

def project_Xmat(X_mat, proj_file):
    ''' Projects the Orthonormal Matrix in the images matrix to reduce dimensionality. '''
    orthMat = hdf5storage.loadmat(proj_file)
    idx_name = list(orthMat)[0]
    orthMat = orthMat[idx_name].astype('float')
    X_proj  = np.dot(X_mat,orthMat)
    return X_proj

def process_image(img_path,param):
    '''Execute all preprocessing steps per image. This function can be called
    to train the model and to production environment.'''
    # resize the image to pattern
    img = resize_image(img_path,param.szimg) 
    # gray scale - just one color channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # minmax normalization
    img = normalize(img) # PROBLEM CONVERT UINT8 TO FLOAT64...
    # contour mask/filter
    img = apply_mask(img, param.filtertype)
    # slidding window and vetorization
    img = slidding(img,param.pool,param.szslidwin)
    # binarization
    img = img2_bin(img) # tamanho errado, seriam 32bits?
    return img

def to_process(path,prefix,proj_path,param, y_val='None'):
    '''Takes images by folder to get the correct class and generate the set 
    to train the models. '''
    y_mat = 'None'
    if y_val != 'None':
        paths = sorted([f.path for f in os.scandir(path) if f.is_dir()])
        lens_paths = [len([f.path for f in os.scandir(g) if f.is_file()]) for g in paths]
        N = sum(lens_paths)
    #    X_mat = np.zeros(shape=(N, (param.szimg**2)*8*3), dtype='uint8')
        X_mat = np.zeros(shape=(N, (param.szimg**2)*32), dtype='uint8')
        y_mat = np.repeat(np.array(y_val), lens_paths)
        # augmentation
        #augment(path,path,param.nsample)
        i=0
        for folder in paths:
            print("--- Starting - folder: {0} ---".format(os.path.basename(folder)))
            start_time = time.time()
            imgs = [f.path for f in os.scandir(folder) if f.is_file()]
            
            for img_path in imgs:
                X_mat[i,:] = process_image(img_path, param)
                print('\r Processing '+str(((i+1)*100)/N)+'% completos...' )
                i+=1
    
            tot_in_sec = time.time() - start_time
            print("--- {0} seconds - folder: {1} ---".format(str(tot_in_sec), os.path.basename(folder)))
    else:
        X_mat = process_image(path, param)
    
    # Projecting the quasi-Orthonormal matrix in our X to reduce dimensionality and abstraction
    X_proj = project_Xmat(X_mat, proj_path)
    
    if y_val != 'None':
        
        nameout = 'output/'+prefix+param.curr_datetime+'.npz'
        np.savez_compressed(nameout,X_mat=X_proj,y_mat=y_mat)
        
    return X_proj, y_mat

#loaded = np.load('draft/output/RxCov10_02_07_2020_11_14_03.npz')
#loaded.files
#X = loaded['X_mat']
#y = loaded['y_mat']
