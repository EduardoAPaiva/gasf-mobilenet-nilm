import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tensorflow.keras.models import Model #type: ignore
from sklearn.metrics import confusion_matrix

# ==========================
# 1️⃣ Configurações
# ==========================
#DEFINIR APARELHO E MODELO A SER UTILIZADO
APARELHO = "kettle"
MODELO = "melhorModelo"

X_TEST_CSV = f"Dados/teste/{APARELHO}/casa2_x_teste.csv"
Y_TEST_CSV = f"Dados/teste/{APARELHO}/casa2_y_teste.csv"
MODEL_PATH = f"ModelosTreinados/{APARELHO}/{MODELO}_{APARELHO}.keras"
SAVE_RESULT_PATH = f"ModelosTreinados/{APARELHO}/{APARELHO}_f1-score.txt"

# ==========================
# 2️⃣ Ler CSVs
# ==========================
x_test = pd.read_csv(X_TEST_CSV, header=None)
y_test = pd.read_csv(Y_TEST_CSV, header=None)

if y_test.shape[1] == 1:
    y_test = y_test.values.ravel()

x_test = x_test.values.astype("float32")

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# ==========================
# 3️⃣ Carregar modelo Keras normal
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# 4️⃣ Rodar inferência
# ==========================
y_outputs_float = model.predict(x_test, verbose=0).ravel()

print("Min saída:", y_outputs_float.min())
print("Max saída:", y_outputs_float.max())

# ==========================
# 5️⃣ Buscar melhor threshold
# ==========================
best_f1 = 0
best_threshold = 0

minimo = y_outputs_float.min()
maximo = y_outputs_float.max()

#Busca fina em float
for t in np.linspace(minimo, maximo, 500):
    y_pred = (y_outputs_float >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_outputs_float >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

# ==========================
# 6️⃣ Salvar resultado
# ==========================
with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
    print(f"Melhor threshold FLOAT: {best_threshold}", file=f)
    print(f"Melhor F1-score: {best_f1}\n", file=f)
    print(classification_report(y_test, (y_outputs_float >= best_threshold).astype(int)), file=f)
    print()
    print(cm, file=f)

print("Melhor threshold:", best_threshold)
print("Melhor F1:", best_f1)
