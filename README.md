# Classificação de Síndromes Genéticas com Embeddings (KNN)

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
        "1": { "accuracy": [0.85], "f1": [0.82], "auc": [0.87], "top_k": [0.90] },
        "2": { "accuracy": [0.88], "f1": [0.84], "auc": [0.89], "top_k": [0.92] }
        ...
      },
      "cosine": {
        "1": { "accuracy": [0.83], "f1": [0.80], "auc": [0.85], "top_k": [0.88] },
        "2": { "accuracy": [0.86], "f1": [0.83], "auc": [0.88], "top_k": [0.91] }
        ...
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

## **Pré-processamento**

Transformar o arquivo de embeddings (`mini_gm_public_v0.1.p`) em um formato tabular, validando os dados e gerando estatísticas básicas.

Execute o comando abaixo para processar os dados:

```bash
python preprocessing.py --path mini_gm_public_v0.1.p
    --path (str): Caminho do arquivo de embeddings (padrão: mini_gm_public_v0.1.p).
```

## **Visualização**

Reduzir a dimensionalidade dos embeddings para **2D** usando **t-SNE** e analisar visualmente os agrupamentos das síndromes.

```bash
python visualization.py --path mini_gm_public_v0.1_processed.p --n_clusters 10
    --path (str): Caminho do arquivo de embeddings processado (padrão: mini_gm_public_v0.1_processed.p)
    --n_clusters (int): Número de clusters para K-Means (padrão: 10)
```

## Classificação
Para treinar e avaliar o modelo, execute:
```bash
python classification.py --path mini_gm_public_v0.1_processed.p --top_k 2
    --path (str): Caminho do arquivo de embeddings processado (padrão: mini_gm_public_v0.1_processed.p)
    --top_k (int): Valor de K para Top-K Accuracy (padrão: 2)
```

## Métricas
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

<table>
  <tr>
    <td align="center"><a href="https://github.com/MatMB115"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/63670910?v=4" width="100px;" alt=""/><br /><sub><b>Matheus Martins</b></sub></a><br /><a href="https://github.com/MatMB115/" title="RepiMe">:technologist:</a></td>
  </tr>
</table>