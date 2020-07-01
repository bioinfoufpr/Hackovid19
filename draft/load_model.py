#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:51:29 2020

@author: datascience
"""

from joblib import load
import RXcovid as rx

vars_prod = ['imgs_size', 'proj_path', 'path_img', 'model_path']
locals_var =  locals()

if not(all([var in locals_var for var in vars_prod])):
    imgs_size = int(input("Tamanho para resize da imagem: "))
    proj_path = input("Caminho completo da matrix ortonormal para projeção: ")

path_img  = input("Nome da imagem  (nome completo): ") # where do the images comes from?
model_path = input("Caminho e nome do modelo: ")

prefix = ''
    
parameters = rx.TransfParam(2,imgs_size,'maxpool','none',3)
X,y = rx.to_process(path_img,prefix,proj_path,parameters)

model = load(model_path)
prediction = int(model.predict(X.reshape(1, -1)))
classes = ['Sem detecção', 'Possível COVID-19', 'Possível Pneumonia']

print()
print('A imagem informada foi classificada como: {0}'.format(classes[prediction]) )