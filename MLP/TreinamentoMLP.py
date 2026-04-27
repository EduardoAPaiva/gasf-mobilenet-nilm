import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

#DEFINE O APARELHO A TREINAR O MODELO DE MLP
APARELHO = "kettle"

# ==========================
# 1️⃣ Ler CSV
# ==========================

def read_csv_safe(path, header=None):
    """Lê CSV se existir e não estiver vazio, caso contrário retorna DataFrame vazio"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path, header=header)
        if df.empty:
            return pd.DataFrame()  # CSV existe mas está vazio
        return df
    else:
        return pd.DataFrame()      # arquivo não existe ou tamanho 0

# ==========================
# Treinamento
# ==========================
x_train_casa1 = read_csv_safe(f"Dados/treinamento/{APARELHO}/casa1_x_train.csv", header=None)
x_train_casa5 = read_csv_safe(f"Dados/treinamento/{APARELHO}/casa5_x_train.csv", header=None)
y_train_casa1 = read_csv_safe(f"Dados/treinamento/{APARELHO}/casa1_y_train.csv", header=None)
y_train_casa5 = read_csv_safe(f"Dados/treinamento/{APARELHO}/casa5_y_train.csv", header=None)

# ==========================
# Validação
# ==========================
x_valid_casa1 = read_csv_safe(f"Dados/validacao/{APARELHO}/casa1_x_valid.csv", header=None)
x_valid_casa5 = read_csv_safe(f"Dados/validacao/{APARELHO}/casa5_x_valid.csv", header=None)
y_valid_casa1 = read_csv_safe(f"Dados/validacao/{APARELHO}/casa1_y_valid.csv", header=None)
y_valid_casa5 = read_csv_safe(f"Dados/validacao/{APARELHO}/casa5_y_valid.csv", header=None)


x_train = pd.concat([x_train_casa1, x_train_casa5], axis=0)
y_train = pd.concat([y_train_casa1, y_train_casa5], axis=0)
x_valid = pd.concat([x_valid_casa1, x_valid_casa5], axis=0)
y_valid = pd.concat([y_valid_casa1, y_valid_casa5], axis=0)

x_train = x_train.values
x_valid = x_valid.values

y_train = y_train.values.ravel()
y_valid = y_valid.values.ravel()

# ==========================
# 6️⃣ Criar modelo MLP
# ==========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(x_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

#Checkpoint que salva o modelo com melhor val_accuracy
checkpoint = ModelCheckpoint(
    filepath=f"ModelosTreinados/{APARELHO}/melhorModelo_{APARELHO}.keras",
    monitor="val_accuracy",          
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# ==========================
# 8️⃣ Treinar
# ==========================
history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint],
    verbose=1
)

# ==========================
# 9️⃣ Avaliar
# ==========================
loss, acc = model.evaluate(x_valid, y_valid)
print(f"Acurácia validação: {acc:.4f}")
