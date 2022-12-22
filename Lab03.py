# -*- coding: utf-8 -*-
# Imagens Biomédicas 2021.1
# Laboratório 3 - Fabio Kenji Makita 120369

#%% 0. Bibliotecas utilizadas

import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV
import skimage
import skimage.exposure
import scipy.signal

#%% 1. Criação de dois vetores

w = np.array ([1, 2, 3, 2, 8]) # Kernel
f = np.array ([0, 0, 0, 1, 0, 0, 0, 0]) # Sinal
pad = np.zeros ((1,4), int)
fpadding = np.concatenate((pad, f, pad), axis = None)

#%% 2. Questão 02

(Lo,) = np.shape(f)
(Lw,) = np.shape(w)
(L,) = np.shape(fpadding) # L = Lo + 2*(Lw - 1)

for i in range(L):
    fpadding[i]
    
#%% 3. Questão 03
# Correlação realizada manualmente

LfullCorr = Lo + (Lw - 1)
cor = np.zeros(LfullCorr, int)
for i in range(LfullCorr):
    cor[i] = np.sum(w[0:Lw]*fpadding[i:i+Lw])

#%% 4. Questão 04
# Correlação feita por função

corFuncao = np.correlate(f,w,"full")

#%% 5. Questão 05
# Corte realizado manualmente

ccrop = np.zeros(8, int)
ccrop[0:7] = cor[2:9]

#%% 6. Questão 06
# Máscara 3x3

f = np.zeros((5, 5), float) # Sinal
f[2,2] = 1
w = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # Kernel
(M,N) = np.shape(f) # M linhas e N colunas

#%% 7. Questão 07
# Correlação da Matriz f e Máscara w

cor2 = np.zeros((3,3), float)
for lin in range(3):
    for col in range(3):
        #cor[lin, col] = 1
        cor2[lin, col] = np.sum((w[0:3, 0:3]*f[lin:lin+3, col:col+3]))

#%% 8. Questão 08 - Desafio 4 for
"""
cor2 = np.zeros((3, 3), float)

for lin in range(3): # Varredura das linhas do SINAL
    for col in range(3): # Varredura das colunas do SINAL
        for a in range(3): # Varredura das linhas do KERNEL
            for b in range(3): # Varredura das colunas do KERNEL
                cor2[lin, cor] = np.sum(f[lin+1:lin+4, col+1:col+4] * w[a:a+3, b:b+3]) # Correlação
"""
#%% 9. Questão 09
# Correlação utilizando a função

corFuncao2 = scipy.signal.correlate2d(f, w,boundary= 'symm', mode = 'same')

#%% 10. Questão 10
# Leitura e normalização da img

i0 = cv2.imread('Mamography.pgm', 0)
in0 = skimage.img_as_float(i0)

# 10.a) O fator de correlação equivale a quantidade de elementos dentro da matriz de correlação.
w = np.ones((3,3), float)/9
#w = np.ones((9,9), float)/81
#w = np.ones((15,15), float)/255

plt.figure()
plt.title('Image0')
plt.imshow(in0, cmap = 'gray')

#10.b) O efeito de blur acontece devido ao tamanho da máscara
def filtro(largura):
    w = np.ones((int(largura),int(largura)), float)/largura**2
    in0Filt = scipy.signal.correlate2d(in0, w, boundary = 'symm', mode = 'same')
    
    plt.figure()
    plt.title(f'imFilt ' + str(largura) + 'x' +str(largura))
    plt.imshow(in0Filt, cmap = 'gray')

filtro(5)
filtro(9)
filtro(10)
filtro(15)
filtro(30) # Teste
filtro(50) # Teste

#10.c) É possível observar que a imagem sofre blur conforme o aumento da dimensão da máscara.