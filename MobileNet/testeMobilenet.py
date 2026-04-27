import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#DEFINA O APARELHO E O MODELO QUE DESEJA TESTAR
APARELHO = "fridge"
MODELO = "melhorModelo"

# ===== CONFIGURAÇÕES =====
SAVE_RESULT_PATH = f"ModelosTreinados/{APARELHO}/result_{MODELO}.txt"
MODEL_PATH = f"ModelosTreinados/{APARELHO}/{MODELO}_{APARELHO}.keras"  # caminho do .keras
IMAGE_PATH_ON = f"gaf_images/teste/{APARELHO}/on"      # imagem para testar
IMAGE_PATH_OFF = f"gaf_images/teste/{APARELHO}/off"
IMG_SIZE = 128                        # tamanho usado no treino
CLASS_NAMES = [
    "off",
    "on",
]  # ajuste para suas classes


# ===== CARREGAR MODELO =====
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado com sucesso!")

# ===== FUNÇÃO DE PRÉ-PROCESSAMENTO =====

def preprocess_image(image_path):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img = tf.keras.utils.img_to_array(img)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img

#=====================================================================================

y_true = []
y_probs = []

contador = 0
Porcentagem = 0

imagensOn = os.listdir(IMAGE_PATH_ON)
imagensOff = os.listdir(IMAGE_PATH_OFF)

qtdImagens = len(imagensOn) + len(imagensOff)

print(f"\n========== {APARELHO} ==========")
print(f"QUANTIDADE DE IMAGENS: {qtdImagens}")

# ========== Imagens ON ==========

for filename in imagensOn:
    # ===== CARREGAR E PROCESSAR IMAGEM =====
    try:
        img = preprocess_image(IMAGE_PATH_ON + "/" + filename)
    except:
        continue

    pred = model.predict(img, verbose=0)[0][0]

    contador += 1
    progresso = (contador / qtdImagens) * 100
    if progresso >= Porcentagem + 5:
        Porcentagem += 5
        print(f'Total de imagens: {contador}, Porcentagem: {Porcentagem}%')

    y_true.append(1)
    y_probs.append(pred)
    


# ========== Imagens OFF ==========

for filename in imagensOff:
    # ===== CARREGAR E PROCESSAR IMAGEM =====
    try:
        img = preprocess_image(IMAGE_PATH_OFF + "/" + filename)
    except:
        continue

    pred = model.predict(img, verbose = 0)[0][0]

    contador += 1
    progresso = (contador / qtdImagens) * 100
    if progresso >= Porcentagem + 5:
        Porcentagem += 5
        print(f'Total de imagens: {contador}, Porcentagem: {Porcentagem}%')

    y_true.append(0)
    y_probs.append(pred)

# ========== Calculo do threshold ótimo ==========

best_f1 = 0
best_threshold = 0

#testa diversos threshold diferentes
for t in np.arange(0.0, 1.01, 0.01):
    y_pred = (y_probs >= t).astype(int)
    f1 = f1_score(y_true, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred_final = (y_probs >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred_final)

with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:

    print(f"Melhor threshold: {best_threshold}", file=f)
    print(f"Melhor F1-score: {best_f1}\n", file=f)
    print(classification_report(y_true, y_pred_final), file=f)
    print("", file=f)
    print(cm, file=f)
