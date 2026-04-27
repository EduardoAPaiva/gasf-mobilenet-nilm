# GASF MobileNet NILM

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

## 🧠 Pipeline Proposto

O pipeline do projeto segue as seguintes etapas:

1. **Aquisição do sinal elétrico**
   Dados agregados de consumo (potência ativa, potência aparente e tensão), utilizando a base de dados UK-DALE. A partir da lei de ohm, obtém-se os valores de potência reativa e corrente

2. **Segmentação temporal**
   Divisão do sinal em janelas de diferentes tamanhos, a depender da carga analisada

3. **Transformação para imagem**
   Aplicação de técnicas como Gramian Angular Summation Field (GASF) para converter séries temporais em imagens utilizando os canais RGB da imagem, onde cada canal representa uma grandeza elétrica diferente (respectivamente: potência ativa, potência reativa e corrente)

4.  **Transformação para vetor**
   O sinal obtido possui três canais. Para a utilização do MLP, os três canais são concatenados em um único vetor, seguindo a mesma ordem apresentada no item 3

5. **Treinamento dos modelos**

   * MobileNet (rede profunda, otimizada para eficiência)
   * MLP (rede rasa para comparação)

6. **Classificação da carga**
   Identificação do aparelho (ex: geladeira, micro-ondas, etc.)

7. **Avaliação de desempenho**
   Cada modelo treinado é avaliado utilizando duas métricas de avaliação, sendo elas a acurácia e o *f1-score*

---

### 💡 Motivação

A transformação de séries temporais em imagens permite o uso de arquiteturas de visão computacional (CNNs), enquanto o uso de MobileNet torna o sistema viável para execução em dispositivos embarcados de baixo custo.


---

## 📂 Estrutura do projeto

```
.
├── MLP/
│   ├── dadosCasa1.ipynb
│   ├── dadosCasa5.ipynb
│   ├── testeCasa2.ipynb
│   ├── treinamentoMLP.py
│   └── testeMLP.py
│
├── MobileNet/
│   ├── treinamentoMobilenet.py
│   ├── testeMobilenet.py
│   ├── ImagensTesteCasa2.ipynb
│   ├── ImagensTreinamentoCasa1.ipynb
│   ├── imagensTreinamentoCasa5.ipynb
│   └── exampleimages/
│
└── README.md
```

---

### 🧠 Organização

* **MLP/**
  Contém os scripts e notebooks relacionados ao modelo de rede neural rasa (MLP), incluindo:

  * preparação dos dados
  * treinamento
  * teste
  * avaliação

* **MobileNet/**
  Contém os scripts e notebooks relacionados ao modelo profundo (MobileNet), incluindo:

  * geração de imagens (GASF)
  * treinamento da CNN
  * avaliação

---

### 💡 Observação

A separação em duas pastas permite comparar diretamente abordagens:

* **Deep Learning (MobileNet)**
* **Shallow Learning (MLP)**

mantendo o pipeline organizado e modular.

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

Ambos arquivos de testes das redes neurais (testeMobilenet.py e testeMLP.py) calculam o melhor threshold para o modelo e salva as informações de acurácia e f1-score  em um arquivo .txt salvo juntamente do modelo treinado.

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
