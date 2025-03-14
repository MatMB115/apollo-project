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

- O arquivo principal de dados (`mini_gm_public_v0.1.p`) contém um dicionário cujos níveis hierárquicos representam `syndrome_id`, `subject_id` e `image_id`, mapeando cada imagem ao respectivo embedding de 320 dimensões.
- A estrutura geral:
```json
{ 'syndrome_id': 
    { 'subject_id': 
        { 'image_id': [320-dimensional embedding] } 
    } 
}
```

## Requisitos
-Python 3.13.2
  - Numpy
  - Pandas
  - 