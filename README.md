# (pt-br) Classificação de Síndromes Genéticas com Embeddings  com KNN

Repositório desenvolvido para armazenar o projeto de classificação de síndromes genéticas a partir de embeddings de imagens. É possível encontrar instruções iniciais sobre como configurar o ambiente, organizar os dados e executar os scripts de pré-processamento, visualização e classificação.

Para sanar eventuais dúvidas entre em contato pelo meu e-mail: [matheusmbatista2009@gmail.com](matheusmbatista2009@gmail.com)

---

## Visão Geral

Este projeto consiste em analisar embeddings (vetores de 320 dimensões) derivados de imagens de pacientes com diferentes síndromes genéticas. O objetivo é:

1. **Pré-processar** os dados e preparar uma estrutura limpa e coerente para análise.  
2. **Visualizar** as embeddings em 2D usando t-SNE, facilitando a identificação de padrões e clusters.  
3. **Classificar** corretamente as imagens em suas respectivas síndromes usando o algoritmo KNN (com diferentes métricas de distância).  
4. **Avaliar** o desempenho do modelo com métricas como AUC, F1-Score, Top-k Accuracy, entre outras.  

---

## Estrutura dos Dados

### Dataset proposto
- O arquivo principal de dados (`mini_gm_public_v0.1.p`) contém um dicionário cujos níveis hierárquicos representam `syndrome_id`, `subject_id` e `image_id`, mapeando cada imagem ao respectivo embedding de 320 dimensões.
- A estrutura geral:
```json
{ 'syndrome_id': 
    { 'subject_id': 
        { 'image_id': [320-dimensional embedding] } 
    } 
}
```

### **Arquivos Gerados**
- **`mini_gm_public_v0.1_processed.p`**  
  - Arquivo pickle contendo os dados pré-processados em formato tabular.  
  - Estrutura:
    ```json
    [
      {
        "syndrome_id": "S1",
        "subject_id": "Sub1",
        "image_id": "Img1",
        "embedding": [320-dimensional embedding]
      },
      ...
    ]
    ```
  - Utilizado para visualização e classificação.

- **`knn_detailed_results.p`**  
  - Arquivo pickle contendo os resultados detalhados da classificação KNN.  
  - Estrutura:
    ```json
    {
      "euclidean": {
        "1": { "accuracy": [0.85], "f1": [0.82], "auc": [0.87], "top_k": [0.90], "y_true": [...], "y_proba": [...]}, // cada métrica 10 vezes (folds)
        "2": { "accuracy": [0.88], "f1": [0.84], "auc": [0.89], "top_k": [0.92], "y_true": [...], "y_proba": [...] },
        ...
        "15": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] }
      },
      "cosine": {
        "1": { "accuracy": [0.83], "f1": [0.80], "auc": [0.85], "top_k": [0.88], "y_true": [...], "y_proba": [...] },
        "2": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] },
        ...
        "15": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] }
      },
      "top_k": 2
    }
    ```
  - Utilizado na etapa de avaliação e geração de métricas.

## Configuração de Ambiente
Este projeto foi desenvolvido e testado em **Python 3.13**.

### **Instalação das Dependências**
Execute o seguinte comando para instalar os pacotes necessários:

```bash
pip install -r requirements.txt
```
## Pipeline
O pipeline inclui **pré-processamento dos dados, visualização, classificação com KNN e avaliação de métricas**.

O pipeline pode ser executado **de duas formas**:

1. **Execução completa** através do `main.py`, que realiza todas as etapas do projeto automaticamente.
2. **Execução por etapas**, chamando cada script individualmente.

---

### Execução Completa - [main.py](main.py)

Para rodar todo o pipeline automaticamente, utilize:

```bash
python main.py
```

Este script suporta vários argumentos de linha de comando que permitem ao usuário configurar diferentes aspectos do **processamento de dados, visualização, classificação e avaliação de métricas**.

#### Parâmetros da Linha de Comando

| Argumento              | Tipo    | Valor Padrão                      | Descrição                                                                                         |
| ---------------------- | ------- | --------------------------------- | ------------------------------------------------------------------------------------------------- |
| `--input_pickle`       | `str`   | `mini_gm_public_v0.1.p`           | Caminho para o arquivo pickle original contendo os embeddings brutos.                             |
| `--processed_pickle`   | `str`   | `mini_gm_public_v0.1_processed.p` | Caminho de saída para salvar o arquivo pré-processado.                                            |
| `--output_knn_results` | `str`   | `knn_detailed_results.p`          | Caminho onde os resultados detalhados da classificação KNN serão armazenados.                     |
| `--tsne_perplexity`    | `float` | `30.0`                            | Parâmetro de perplexidade para t-SNE, que equilibra os aspectos locais e globais na visualização. |
| `--tsne_seed`          | `int`   | `115`                             | Semente aleatória para t-SNE, garantindo reprodutibilidade dos resultados.                        |
| `--n_clusters`         | `int`   | `10`                              | Número de clusters usados no K-Means para a visualização dos embeddings.                          |
| `--top_k`              | `int`   | `2`                               | Valor de K para **Top-K Accuracy** na classificação KNN.                                          |
| `--kfold_seed`         | `int`   | `115`                             | Semente aleatória para a validação cruzada KFold.                                                 |
| `--k_neighbors`        | `int`   | `15`                              | Número de vizinhos (`K`) utilizados na classificação K-Nearest Neighbors (KNN).                   |

---

#### Exemplo de Uso

Execute o script usando o seguinte comando:

```bash
python main.py --input_pickle my_data.p --tsne_perplexity 40 --n_clusters 8 --k_neighbors 10
```

### Pré-processamento - [preprocessing.py](preprocessing.py)

Transformar o arquivo de embeddings (`mini_gm_public_v0.1.p`) em um formato tabular, validando os dados e gerando estatísticas básicas.

Execute o comando abaixo para processar os dados:

```bash
python preprocessing.py --path mini_gm_public_v0.1.p
    --path (str): Caminho do arquivo de embeddings (padrão: mini_gm_public_v0.1.p).
```

### Visualização - [visualization.py](visualization.py)

Reduzir a dimensionalidade dos embeddings para **2D** usando **t-SNE** e analisar visualmente os agrupamentos das síndromes.

```bash
python visualization.py --path mini_gm_public_v0.1_processed.p --n_clusters 10
    --path (str): Caminho do arquivo de embeddings processado (padrão: mini_gm_public_v0.1_processed.p)
    --n_clusters (int): Número de clusters para K-Means (padrão: 10)
```

### Classificação - [classification.py](classification.py)
Para treinar e avaliar o modelo, execute:
```bash
python classification.py --path mini_gm_public_v0.1_processed.p --top_k 2
    --path (str): Caminho do arquivo de embeddings processado (padrão: mini_gm_public_v0.1_processed.p)
    --top_k (int): Valor de K para Top-K Accuracy (padrão: 2)
```

### Métricas - [metrics_eval](metrics_eval.pys)
Após a classificação, o modelo é avaliado utilizando métricas como **AUC (Área sob a Curva ROC)**, **F1-Score**, **Top-K Accuracy** e **Accuracy**.

```bash
python metrics_eval.py --path knn_detailed_results.p
    --path (str): Caminho do arquivo de resultados da classificação (padrão: knn_detailed_results.p)
```

## Requisitos
- Python 3.13.2
  - Numpy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn 

---

# (en-us) Genetic Syndrome Classification with Embeddings with KNN

Repository developed to store the project for classifying genetic syndromes from image embeddings. Initial instructions on how to set up the environment, organize the data, and run preprocessing, visualization, and classification scripts can be found here.

For any questions, contact me via email: [matheusmbatista2009@gmail.com](matheusmbatista2009@gmail.com)


## Overview

This project analyzes embeddings (320-dimensional vectors) derived from images of patients with different genetic syndromes. The objectives are:

1. **Preprocess** the data and prepare a clean and coherent structure for analysis.
2. **Visualize** the embeddings in 2D using t-SNE, facilitating the identification of patterns and clusters.
3. **Classify** images into their respective syndromes using the KNN algorithm (with different distance metrics).
4. **Evaluate** model performance with metrics such as AUC, F1-Score, Top-k Accuracy, among others.


## Data Structure

### Proposed Dataset
- The main data file (`mini_gm_public_v0.1.p`) contains a dictionary whose hierarchical levels represent `syndrome_id`, `subject_id`, and `image_id`, mapping each image to the respective 320-dimensional embedding.
- General structure:

```json
{ 'syndrome_id':
    { 'subject_id':
        { 'image_id': [320-dimensional embedding] }
    }
}
```

### **Generated Files**
- **`mini_gm_public_v0.1_processed.p`**  
  - Pickle file containing the preprocessed data in tabular format.
  - Structure:

    ```json
    [
      {
        "syndrome_id": "S1",
        "subject_id": "Sub1",
        "image_id": "Img1",
        "embedding": [320-dimensional embedding]
      },
      ...
    ]
    ```
  - Used for visualization and classification.

- **`knn_detailed_results.p`**  
  - Pickle file containing detailed KNN classification results.  
  - Structure:

    ```json
    {
      "euclidean": {
        "1": { "accuracy": [0.85], "f1": [0.82], "auc": [0.87], "top_k": [0.90], "y_true": [...], "y_proba": [...]}, // each metric 10 times (folds)
        "2": { "accuracy": [0.88], "f1": [0.84], "auc": [0.89], "top_k": [0.92], "y_true": [...], "y_proba": [...] },
        ...
        "15": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] }
      },
      "cosine": {
        "1": { "accuracy": [0.83], "f1": [0.80], "auc": [0.85], "top_k": [0.88], "y_true": [...], "y_proba": [...] },
        "2": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] },
        ...
        "15": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91], "y_true": [...], "y_proba": [...] }
      },
      "top_k": 2
    }
    ```
  - Used in the evaluation stage and metric generation.

## Environment Setup

This project was developed and tested in **Python 3.13**.

### **Installing Dependencies**
Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Pipeline
The pipeline includes **data preprocessing, visualization, classification with KNN, and metric evaluation**.

The pipeline can be executed **in two ways**:

1. **Full execution** via `main.py`, which automatically performs all project stages.
2. **Step-by-step execution**, running each script separately.


### **Full Execution - [main.py](main.py)**

To automatically run the entire pipeline, use:

```bash
python main.py
```

This script supports multiple command-line arguments, allowing users to configure various aspects of **data processing, visualization, classification, and metric evaluation**. More info about the arguments in Table [1](#parâmetros-da-linha-de-comando).
### **Step-by-Step Execution**

#### **Preprocessing - [preprocessing.py](preprocessing.py)**

```bash
python preprocessing.py --path mini_gm_public_v0.1.p
```

#### **Visualization - [visualization.py](visualization.py)**

```bash
python visualization.py --path mini_gm_public_v0.1_processed.p --n_clusters 10
```

#### **Classification - [classification.py](classification.py)**

```bash
python classification.py --path mini_gm_public_v0.1_processed.p --top_k 2
```

#### **Metrics Evaluation - [metrics_eval.py](metrics_eval.py)**

```bash
python metrics_eval.py --path knn_detailed_results.p
```

## Requirements

- Python 3.13.2
  - Numpy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn

---

<table>
  <tr>
    <td align="center"><a href="https://github.com/MatMB115"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/63670910?v=4" width="100px;" alt=""/><br /><sub><b>Matheus Martins</b></sub></a><br /><a href="https://github.com/MatMB115/" title="RepiMe">:technologist:</a></td>
  </tr>
</table>