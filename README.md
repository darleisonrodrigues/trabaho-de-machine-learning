# 🔬 Projeto de Redução de Dimensionalidade e Clustering - Machine Learning

Projeto organizado para as **Atividades de Redução de Dimensionalidade (Etapa 1.1) e Clustering (Etapa 1.2)** usando **Paradigma Não-Supervisionado de Machine Learning** com imagens de faces.

## 📋 Objetivos

### Etapa 1.1 - Redução de Dimensionalidade
Aplicar e comparar diferentes métodos de redução de dimensionalidade em um dataset de imagens de faces (128x120 pixels, 15.360 features):

- **PCA** (Principal Component Analysis) - 90%, 80%, 75% da variância
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding) - visualização 2D
- **UMAP** (Uniform Manifold Approximation and Projection) - 3, 15, 55, 101 dimensões

### Etapa 1.2 - Clustering
Aplicar algoritmos de clustering nos embeddings gerados e avaliar qualidade:

- **K-means** e **K-medoids** para diferentes valores de K (2-25)
- **Índice de Dunn** para avaliação da qualidade dos clusters
- **Visualizações comparativas** e análise de resultados

## 🗂️ Estrutura do Projeto

```
ML/
├── data/
│   ├── raw/
│   │   └── RecFac/             # Dataset RecFac (640 imagens em 20 subpastas)
│   └── processed/              # Arquivos intermediários (npy, scaler, etc.)
├── outputs/
│   ├── embeddings/             # Projeções salvas (CSV/NPY)
│   ├── figures/               # Gráficos redução dimensionalidade (PNG/HTML)
│   └── clustering/            # Resultados de clustering
│       ├── results/           # Arquivos JSON com métricas
│       └── figures/           # Visualizações de clusters e dashboards
├── src/                       # Scripts Python
│   ├── dataset.py             # Carregamento e processamento de imagens
│   ├── preprocess.py          # Padronização StandardScaler
│   ├── reduce_pca.py          # Análise PCA
│   ├── reduce_tsne.py         # Análise t-SNE
│   ├── reduce_umap.py         # Análise UMAP
│   ├── visualize.py           # Visualizações genéricas
│   ├── clustering.py          # K-means, K-medoids, índice de Dunn
│   └── visualize_clustering.py # Visualizações de clustering
├── main.py                   # Script principal (pipeline completo)
├── clustering_analysis.py    # Script específico para clustering
├── requirements.txt          # Dependências
├── .gitignore               # Arquivos ignorados pelo Git
└── README.md                # Esta documentação
```

## 🛠️ Instalação e Configuração

### 1. Pré-requisitos

- Python 3.8+
- Ambiente virtual (recomendado)

### 2. Instalar Dependências

```bash
# Ativar ambiente virtual
.\venv\Scripts\Activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Dataset RecFac

O dataset RecFac já está colocado em `data/raw/RecFac/` com a seguinte estrutura:

```
data/raw/RecFac/
├── an2i/
│   ├── an2i_left_angry_open.png
│   ├── an2i_left_happy_open.png
│   └── ... (32 imagens por pessoa)
├── at33/
├── boland/
├── bpm/
└── ... (20 pessoas total)
```

**Características do dataset:**
- 20 pessoas diferentes
- ~32 imagens por pessoa (640 imagens total)
- Variações: direção (left/right/straight/up), emoção (angry/happy/neutral/sad), óculos (open/sunglasses)

## 🚀 Como Usar

### Execução Completa (Recomendado)

```bash
python main.py
```

Este comando executa todo o pipeline:
1. Carrega e processa imagens (128x120 grayscale)
2. Aplica padronização (StandardScaler)
3. Executa PCA (95%, 90%, 80%, 75% variância)
4. Executa t-SNE (perplexity 5, 30, 50)
5. Executa UMAP (3, 15, 55, 101 dimensões)
6. Gera visualizações comparativas
7. **Executa clustering (K-means e K-medoids)**
8. **Calcula índices de Dunn**
9. **Gera dashboards comparativos de clustering**

### Execução Apenas de Clustering

Se você já tem os embeddings gerados, pode executar apenas o clustering:

```bash
python clustering_analysis.py
```

### Execução Modular

Você também pode executar cada etapa separadamente:

```bash
# 1. Processar dataset
python src/dataset.py

# 2. Padronizar dados
python src/preprocess.py

# 3. Análise PCA
python src/reduce_pca.py

# 4. Análise t-SNE
python src/reduce_tsne.py

# 5. Análise UMAP
python src/reduce_umap.py

# 6. Visualizações
python src/visualize.py
```

## 📊 Resultados Gerados

### Embeddings (CSV)
- `pca_95percent.csv` - PCA com 95% da variância
- `pca_90percent.csv` - PCA com 90% da variância
- `pca_80percent.csv` - PCA com 80% da variância
- `pca_75percent.csv` - PCA com 75% da variância
- `tsne_perp30_with_pca_embedding.csv` - t-SNE 2D
- `umap_3d.csv` - UMAP 3 dimensões
- `umap_15d.csv` - UMAP 15 dimensões
- `umap_55d.csv` - UMAP 55 dimensões
- `umap_101d.csv` - UMAP 101 dimensões

### Visualizações Estáticas (PNG)
- `pca_variance_explained.png` - Gráfico de variância explicada por componente
- `pca_comparison.png` - Comparação visual dos PCAs
- `tsne_perp30_with_pca_plot.png` - Visualização t-SNE
- `umap_3d_plot.png` - UMAP 3D
- `embeddings_comparison.png` - Comparação geral de métodos

### Visualizações Interativas (HTML)
- `pca_interactive.html` - PCA interativo
- `tsne_perp30_with_pca_interactive.html` - t-SNE interativo
- `umap_3d_interactive.html` - UMAP 3D interativo

## 🔬 Detalhes Técnicos

### Processamento de Imagens
- Conversão para grayscale
- Redimensionamento para 128x120 pixels
- Normalização para [0, 1]
- Achatamento em vetores de 15.360 features

### Padronização
- StandardScaler (z-score normalization)
- Média = 0, Desvio padrão = 1

### Parâmetros dos Algoritmos
- **PCA**: `random_state=42`
- **t-SNE**: `perplexity=[5,30,50]`, `n_iter=1000`, `random_state=42`
- **UMAP**: `n_neighbors=15`, `min_dist=0.1`, `random_state=42`

## 📈 Interpretação dos Resultados

### Redução de Dimensionalidade

#### PCA
- Analise a variância explicada acumulada
- Compare o número de componentes necessárias
- Use para redução dimensional preservando informação

#### t-SNE
- Observe agrupamentos visuais nas projeções 2D
- Identifique separabilidade entre classes
- Use para definir número de clusters para K-means/K-medoids

#### UMAP
- Compare diferentes dimensionalidades
- Analise preservação de estrutura local e global
- Escolha dimensão baseada no trade-off informação/complexidade

### Clustering

#### Índice de Dunn
- **Valores maiores indicam melhor clustering**
- Fórmula: `min(distância entre clusters) / max(distância intra-cluster)`
- Use para comparar diferentes valores de K e algoritmos

#### Resultados Principais (Exemplo)
```
🏆 MELHORES RESULTADOS GLOBAIS:
KMEANS: UMAP 15D com K=22 (Dunn: 1.5758)
KMEDOIDS: UMAP 15D com K=3 (Dunn: 0.4011)

💡 RECOMENDAÇÕES:
• Para baixa dimensionalidade: UMAP 3D
• Para alta dimensionalidade: UMAP 15D
• Algoritmo com melhor performance média: K-means
```

#### Visualizações Geradas
- **Gráficos de Dunn**: Comparação de qualidade por K
- **Método do Cotovelo**: Análise de inertia
- **Heatmaps**: Comparação entre embeddings
- **Clusters 2D/3D**: Visualização dos agrupamentos

## 🎯 Próximos Passos

1. **Análise Visual**: Examine os gráficos gerados para identificar padrões
2. **Interpretação de Clustering**: Analise os índices de Dunn e escolha melhores K
3. **Comparação de Métodos**: Compare eficácia entre PCA, t-SNE e UMAP
4. **Validação**: Compare separabilidade visual com métricas quantitativas
5. **Otimização**: Ajuste hiperparâmetros baseado nos resultados
6. **Relatório**: Use dashboards interativos para apresentação de resultados

## 📦 Dependências

- `numpy>=1.21.0` - Computação numérica
- `pandas>=1.3.0` - Manipulação de dados
- `scikit-learn>=1.0.0` - Algoritmos ML (PCA, t-SNE, StandardScaler)
- `matplotlib>=3.4.0` - Visualizações estáticas
- `plotly>=5.0.0` - Visualizações interativas
- `umap-learn>=0.5.0` - Algoritmo UMAP
- `pillow>=8.0.0` - Processamento de imagens
- `opencv-python>=4.5.0` - Visão computacional
- `tqdm>=4.60.0` - Barras de progresso
- `joblib>=1.0.0` - Serialização de objetos

## 🚨 Observações Importantes

- Os arquivos em `data/` e `outputs/` não são versionados (ver `.gitignore`)
- Todos os algoritmos usam `random_state=42` para reprodutibilidade
- O processamento pode demorar dependendo do tamanho do dataset
- Visualizações interativas podem ser grandes (>1MB)

## 🐛 Resolução de Problemas

### Dataset não encontrado
```
❌ ERRO: Diretório data/raw/faces não encontrado!
```
**Solução**: Crie a pasta e coloque as imagens organizadas por subpastas.

### Erro de memória
```
MemoryError: Unable to allocate array
```
**Solução**: Reduza o tamanho do dataset ou use menos componentes PCA primeiro.

### Dependência não encontrada
```
ModuleNotFoundError: No module named 'umap'
```
**Solução**: Execute `pip install -r requirements.txt`

## 📝 Licença

Este projeto é para fins educacionais da disciplina de Machine Learning.

---

**Autor**: Projeto de Redução de Dimensionalidade  
**Data**: Setembro 2025