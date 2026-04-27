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

## 🚀 Como utilizar

O projeto foi desenvolvido de forma **modular**, onde cada script ou notebook deve ser executado de forma independente.

### 📌 Configuração

Em cada arquivo `.py` ou `.ipynb`, existe uma seção no início (cabeçalho) onde você deve definir o **nome do aparelho (load/appliance)** que será utilizado.

Exemplo:

```python
appliance_name = "fridge"
```

---

### ⚙️ Execução dos módulos

Você pode rodar cada etapa separadamente, dependendo do que deseja fazer:

#### 🖼️ Geração de imagens (GASF)

Executa notebooks ou scripts responsáveis por converter séries temporais em imagens:

```bash
python ImagensTreinamentoCasa1.ipynb
```

ou via Jupyter Notebook:

* Abra o arquivo `.ipynb`
* Execute todas as células

---

#### 🧠 Treinamento do modelo

Treina a rede (ex: MobileNet) com base nas imagens geradas:

```bash
python treinamentoMobilenet.py
```

---

#### 🧪 Teste / Avaliação

Executa a inferência e avaliação do modelo:

```bash
python testeMobilenet.py
```

---

### 🧠 Observação importante

* Cada arquivo funciona de forma independente
* Basta alterar o **nome do aparelho no cabeçalho** para reutilizar o pipeline
* Isso permite testar facilmente diferentes cargas elétricas

---

### 📦 Requisitos

Instale as dependências antes de executar:

```bash
pip install -r requirements.txt
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
