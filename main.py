# Importação das bibliotecas
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import numpy as np
from keras.datasets import mnist as objects

(train_img,train_lab),(test_img,test_lab)=objects.load_data()

for i in range(20) : plt.subplot(4, 5, i + 1)

# Exemplo de visualização da imagem utilizada para teste
plt.imshow(train_img[i],cmap="gray_r")
plt.title("Digit : {}".format(train_lab[i]))
plt.subplots_adjust(hspace=0.5)

print("Forma das imagens de treinamento:", train_img.shape) 
print("Testando a forma das imagens:", test_img.shape)

print("Exemplo de visualização da imagem:") 
print(train_img[0])

# Pixel em comparação a intensidade
plt.hist(train_img[0].reshape(784),facecolor="orange") 
plt.title("Pixel em comparação a sua intensidade",fontsize=16) 
plt.ylabel("PIXEL") 
plt.xlabel("Intensidade")
train_img=train_img/255.0
test_img=test_img/255.0

# Visualização da imagem após normalização
print(train_img[0])

# Pixel em comparação a intensidade após normalização
plt.hist(train_img[0].reshape(784),facecolor="orange") 
plt.title("Pixel em comparação a sua intensidade",fontsize=16) 
plt.ylabel("PIXEL") 
plt.xlabel("Intensidade")

# Criação do modelo
from keras.models import Sequential
from keras.layers import Flatten,Dense 
model=Sequential() 
input_layer= Flatten(input_shape=(28,28)) 
model.add(input_layer) 
hidden_layer1=Dense(512,activation="relu") 
model.add(hidden_layer1) 
hidden_layer2=Dense(512,activation="relu") 
model.add(hidden_layer2) 
output_layer=Dense(10,activation="softmax") 
model.add(output_layer)

# Compilação do modelo
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

# Treinamento do modelo (Realizado 10 vezes)
model.fit(train_img,train_lab,epochs=10)

model.save("project.h5")

# Verificação da acurácia do modelo e da sua perda
loss_and_acc=model.evaluate(test_img,test_lab,verbose=2)
print("Teste de perda", loss_and_acc[0])
print("Teste de acurácia", loss_and_acc[1])

# Plota a primeira imagem utilizada no teste e se o teste foi realizado com sucesso
plt.imshow(test_img[0],cmap="gray_r")
plt.title("Actual Value: {}".format(test_lab[0]))  
prediction=model.predict(test_img)
plt.axis("off")

print("Predicted Value: ",np.argmax(prediction[0]))

if(test_lab[0]==(np.argmax(prediction[0]))):
  print("Successful prediction")
else:
  print("Unsuccessful prediction")

# Cria a função para carregamento e tratamento da imagem
from keras.utils import load_img
from keras.utils import img_to_array 

def load_image(filename):
  img = load_img(filename, grayscale=True, target_size=(28, 28))
  img = img_to_array(img) 
  img = img.reshape(1, 28, 28) 
  img = img.astype('float32')     
  img = img / 255.0     
  return img

# Faz a requisição para inserção da imagem
from google.colab import files 

uploaded = files.upload()

# Visualiza a imagem importada
from IPython.display import Image

Image('quatro.jpg', width=250,height=250)

# Aplica o modelo na imagem importa e plota o resultado obtido
img = load_image("quatro.jpg")

digit=model.predict(img) 

print(np.argmax(digit))
