#!/usr/bin/env python
# coding: utf-8

# ## OpenCV Text Detection using EAST Text Detector

# - ESAT Text Dtector é um modelo de rede neural profunda, estado da arte sobre detecção de texto

# #### To run:
# - python east-text-detector.py 
# 
# #### Requirements:
# - OpenCV 4
# - Imutils 0.5.3

# ### 1. Importando pacotes

# In[1]:


from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2


# ### 2. Construindo argumentos de entrada

# - **Importante**: EAST text tem como requisito que a dimensão da imagem de entrada seja múltiplo de 32

# In[22]:


# criando objeto do tipo parser
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--image", type=str, 
                             default = "imagens/hocuspocus.jpeg" , help="path to input image")
argument_parser.add_argument("-e", "--east", 
                             default = "frozen_east_text_detection.pb", type=str, help="path to EAST Text Detector")
argument_parser.add_argument("-mc", "--min_confidence", type=float,
                             default = 0.5, help="minimum probability required to inspect a region")
argument_parser.add_argument("-w", "--width", type=int,
                             default = 320, help="image width (multiple of 32")
argument_parser.add_argument("-he", "--heigth", type=int,
                             default = 320, help="image heigth (multiple of 32)")
argument_parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
arguments = vars(argument_parser.parse_args())


# ### 3. Load image, resize and work with image dimensions

# In[4]:


# carregando imagem
image = cv2.imread(arguments["image"])
# criando cópia
orig = image.copy()
# get nas dimensões da imagem
(H, W) = image.shape[:2]

# set das novas dimensões da imagem e determinando a razão das mudanças de dimensão
(newW, newH) = (arguments['width'], arguments['heigth'])
rW, rH = (W / float(newW)), (H / float(newH))

# redimensionando a imagem com as novas dimensões
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]


# ### 4. Definindo as duas camadas de saída do EAST

# In[15]:


# 1° saída das probabilidades
# 2º coordenadas do caixa delimitadora do texto
layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]


# ### 5. BlobFromImage
# - A subtração média é usada para ajudar a combater as alterações de iluminação nas imagens de entrada em nosso conjunto de dados. Portanto, podemos ver a subtração média como uma técnica usada para ajudar nossas redes neurais convolucionais.
# 
# 

# In[16]:


# carregando o modelo EAST Text Detector
print("carregando o modelo EAST Text Detector...")
net = cv2.dnn.readNet(arguments["east"])

# aplicando subtração média com blobfromimage e obtendo as duas camadas de saída
blob = cv2.dnn.blobFromImage(image = image, 
                             size = (W, H),
                             mean = (123.68, 116.78, 103.94),
                             swapRB = True,
                             crop = False)
# armazenando o tempo inicial
start = time.time()
# aplicando blob 
net.setInput(blob)
# net.forward retorna duas features: 
# 1. A probabilidade de uma determinada região conter texto
# 2. A geometria sobre a caixa delimitador com o texto
(scores, geometry) = net.forward(layerNames)
# armazenando o tempo final
end = time.time()

# Calculando o tempo para detecção do texto
print("Text detection took {} seconds.".format(end-start))


# ### 6. Loop sobre cada um desses valores sobre probabilidade e geometria
# 
# - rects: armazena as coordenadas da caixa delimitadora (x, y) para regiões de texto;
# - confidências: armazena a probabilidade associada a cada uma das caixas delimitadoras em rects.

# In[24]:


# Aquisição do número de linhas e colunas do volume de pontuações
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop sobre o número de linhas
for y in range (0, numRows):
    # extraindo as pontuações (probabilidades), seguidas pela geométria dos
    # dados usados para derivar possíveis coordenadas da caixa delimitadora que envolvem o texto
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    
    # loop sobre o número de colunas
    for x in range (0, numCols):
        # se a probabilidade do score for inferior que o mínimo de confiança, ignoramos
        if scoresData[x] < arguments["min_confidence"]:
            continue
        
        # OBS: EAST Text Detect naturalemente reduz o tamanho da imagem que passará pela RN.
        # calculando o fator de deslocamento, pois nossos mapas de recursos resultantes
        # são 4x menores que a imagem de entrada
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        
        # extraindo o ângulo de rotação para a previsão e depois calcular o seno e o cosseno
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        
        # usando o volume da geometria para derivar a largura e a altura da caixa delimitadora
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        
        # calculando as coordenadas x e y para caixa delimitadora da detecção do texto
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        
        # adicionando a caixa delimitador e a probabilidade as listas rects e confidences
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])


# - A etapa final é aplicar a supressão não máxima às nossas caixas delimitadoras para suprimir caixas delimitadoras fracas e sobrepostas e exibir as previsões de texto resultantes:

# In[25]:


boxes = non_max_suppression(np.array(rects), probs=confidences)
 
# loop sobre cada caixa delimitadora
for (startX, startY, endX, endY) in boxes:
    # corrigindo a caixa delimitadora com a razão das mudanças de dimensão que realizamos anteriormente.
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # desenhando a caixa delimitador na imagem
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    


# In[26]:


cv2.imshow("Text Detection", orig)
cv2.waitKey(0)

