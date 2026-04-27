# GASf MobileNet NILM

🇧🇷 Projeto de iniciação científica focado em Non-Intrusive Load Monitoring (NILM), utilizando redes neurais profundas e rasas para classificação de cargas residenciais em sistemas embarcados de baixo custo.

🇺🇸 Undergraduate research project on Non-Intrusive Load Monitoring (NILM), using deep and shallow neural networks for residential load classification on low-cost embedded systems.

---

## 📌 Overview

Este projeto investiga a viabilidade do uso de modelos de aprendizado de máquina para identificação de cargas elétricas a partir de sinais agregados.

A abordagem inclui:

* Transformação de séries temporais em imagens (GAF/RP)
* Uso de MobileNet (deep learning)
* Comparação com MLP (rede rasa)
* Foco em eficiência computacional para sistemas embarcados

---

## 🧠 Pipeline

1. Aquisição de sinais elétricos
2. Pré-processamento
3. Conversão para imagem (GAF/RP)
4. Treinamento dos modelos
5. Avaliação

---

## 📂 Estrutura do projeto

```
.
├── exampleimages/        # Exemplos de imagens GAF
├── treinamentoMobilenet.py
├── testeMobilenet.py
├── ImagensTesteCasa2.ipynb
├── ImagensTreinamentoCasa1.ipynb
├── imagensTreinamentoCasa5.ipynb
└── README.md
```

---

## ⚙️ Tecnologias

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 🚀 Como executar

```bash
pip install -r requirements.txt
python treinamentoMobilenet.py
```

---

## 📊 Objetivo

Avaliar a eficiência e viabilidade de modelos de classificação de carga em cenários reais, com restrições de hardware.

---

## 📌 Observações

* Dataset não incluído no repositório
* Imagens presentes são apenas exemplos

---

## 👨‍💻 Autor

Eduardo Alves Paiva
