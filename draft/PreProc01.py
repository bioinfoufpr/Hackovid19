#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:54:12 2020

@author: Camila Perico
"""
#import lib.preprocess_image as ppi
import lib.RXcovid as rx

# local of import and saves
pathin  = input("Nome da pasta origem:") # where do the images comes from?
pathout = input("Nome da pasta output:") # where do the images go?
prefix  = input("Prefixo nome dos outputs:") # where do the images go?

# define parameters
parameters = rx.TransfParam(2,250,'maxpool','none',3)

# train the system
rx.ToTrain(pathin,pathout,prefix,parameters)

# use the trained system
#rx.ToUse(pathin,pathout)

import numpy as np
loaded = np.load('out/cov.npz')