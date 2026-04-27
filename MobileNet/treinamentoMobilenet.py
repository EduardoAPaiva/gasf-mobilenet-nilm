import tensorflow as tf
from tensorflow.keras import Model # type: ignore
from tensorflow.keras.applications import MobileNetV3Small # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import os
import shutil

#DEFINA O APARELHO QUE QUER TREINAR A MOBILENET
APARELHO = "fridge"

#CAMINHOS PARA OS DADOS DE TREINAMENTO E VALIDACAO
trainPath = f"gaf_images/treinamento/{APARELHO}"
validPath = f"gaf_images/validacao/{APARELHO}"

#CAMINHO PARA O LOCAL DE SALVAMENTO DO MODELO TREINADO
modelSavedPath = f"ModelosTreinados/{APARELHO}"

#CRIA A PASTA DO LOCAL DE SALVAMENTO
shutil.rmtree(f"ModelosTreinados/{APARELHO}", ignore_errors=True)     #APAGA A PASTA CASO JA EXISTA
os.makedirs(f"ModelosTreinados/{APARELHO}", exist_ok=True)

#Carregando as imagens dos diretorios acima
trainGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(trainPath, target_size=(128, 128), batch_size=32, class_mode='binary')
validGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(validPath, target_size=(128, 128), batch_size=32, class_mode='binary')

#Criando modelo base usando pesos do imagenet
baseModel = MobileNetV3Small(input_shape=(128, 128, 3), weights='imagenet', include_top=False)

#Define os pesos importados da imagenet como congeladas
baseModel.trainable = False

#Definindo os parametros de profundidade
#Adiciona-se duas camadas densas com ativacao relu, essas serao as camadas treinadas no momento
x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#Cria a camada de saida com ativacao sigmoide (saida no intervalo [0,1])
predictionLayer = Dense(1, activation='sigmoid')(x)

#Cria o modelo com os pesos obtidos do imagenet mais as camadas densas adicionadas
model = Model(inputs=baseModel.input, outputs = predictionLayer)

#Compila a mobilenet
optimizer = Adam(learning_rate=0.0001)       #DEFINA AQUI A TAXA DE APRENDIZADO DO TREINAMENTO
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

print("\n========== Modelo carregado com sucesso ==========\n")

#Checkpoint que salva o modelo com o melhor val_accuracy no momento
checkpoint = ModelCheckpoint(
    filepath=f"{modelSavedPath}/melhorModelo_{APARELHO}.keras",
    monitor="val_accuracy",          
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

#Etapa de treinamento com pesos congelados
model.fit(trainGenerator, validation_data=validGenerator, epochs=50, callbacks=[checkpoint])

model = tf.keras.models.load_model(f"{modelSavedPath}/melhorModelo_{APARELHO}.keras")

#Descongela as ultimas 20 camadas
for layer in baseModel.layers[-20:]:
    layer.trainable = True

#Treina novamento com as camadas descongeladas
model.compile(optimizer=Adam(learning_rate=5e-6), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.fit(trainGenerator, validation_data=validGenerator, epochs=20, callbacks=[checkpoint])


#Salvando o modelo treinado
model.save(f"{modelSavedPath}/modeloFinal_{APARELHO}.keras")