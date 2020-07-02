#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:54:12 2020

@author: Camila Perico
"""
#import lib.preprocess_image as ppi
import RXcovid as rx
import run_models as rmdl

from datetime import datetime

# local of import and saves

var_names = ['pathin', 'prefix', 'imgs_size', 'proj_path']

locals_var =  locals()
if not all([val in locals_var for val in var_names]):
    pathin  = input("Pasta para treinamento: ") # where do the images comes from?
    prefix  = input("Prefixo para arquivos de saída: ")
    imgs_size = int(input("Tamanho para resize da imagem: "))
    proj_path = input("Caminho completo da matrix ortonormal para projeção: ")

# define parameters
curr_datetime = datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
parameters = rx.TransfParam(2,imgs_size,'maxpool','none',3,curr_datetime)

# Generate the trainning/test dataset
X, y = rx.to_process(pathin,prefix,proj_path,parameters, y_val=[1,0,2])

# Trainning the model
rmdl.generate_model(X,y,prefix,parameters)