# Configuração do Projeto - Redução de Dimensionalidade
# Parâmetros conforme enunciado da Etapa 3

## Dataset
DATASET_PATH = "data/raw/RecFac"
IMAGE_SIZE = (128, 120)  # Altura x Largura
TOTAL_FEATURES = 15360   # 128 * 120
EXPECTED_IMAGES = 640    # 20 pessoas x ~32 imagens

## Preprocessamento
NORMALIZATION_METHOD = "StandardScaler"  # z-score
RANDOM_STATE = 42  # Seed fixa para reprodutibilidade

## t-SNE (Visualização 2D obrigatória)
TSNE_PARAMS = {
    "n_components": 2,        # SEMPRE 2D conforme enunciado
    "perplexity": 30,         # Valor principal recomendado
    "n_iter": 1000,
    "random_state": 42,
    "use_pca_first": True,    # PCA prévio para estabilidade
    "pca_components": 50      # Componentes PCA prévio
}

## PCA (Variâncias específicas)
PCA_VARIANCE_TARGETS = [0.90, 0.80, 0.75]  # Exatamente conforme enunciado
PCA_PARAMS = {
    "random_state": 42
}

## UMAP (Dimensões específicas)
UMAP_DIMENSIONS = [3, 15, 55, 101]  # Exatamente conforme enunciado
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean",
    "random_state": 42
}

## Saídas esperadas
OUTPUT_FILES = {
    "tsne": "tsne_2d.csv",           # t-SNE 2D
    "pca_90": "pca_var90.csv",       # PCA 90% variância
    "pca_80": "pca_var80.csv",       # PCA 80% variância  
    "pca_75": "pca_var75.csv",       # PCA 75% variância
    "umap_3": "umap_3.csv",          # UMAP 3 dimensões
    "umap_15": "umap_15.csv",        # UMAP 15 dimensões
    "umap_55": "umap_55.csv",        # UMAP 55 dimensões
    "umap_101": "umap_101.csv"       # UMAP 101 dimensões
}

## Visualizações obrigatórias
REQUIRED_PLOTS = [
    "tsne_2d_plot.png",              # t-SNE 2D scatter
    "pca_variance_explained.png",     # Variância acumulada PCA
    "pca_components_count.txt"        # Número de componentes por variância
]

## Estrutura de saída CSV
CSV_COLUMNS = {
    "tsne": ["x", "y", "image_path", "sample_id", "folder", "filename"],
    "pca": ["PC1", "PC2", "...", "PCn", "image_path", "sample_id", "folder", "filename"],
    "umap": ["UMAP1", "UMAP2", "...", "UMAPn", "image_path", "sample_id", "folder", "filename"]
}

## Diretórios
OUTPUTS_DIR = "outputs"
DATA_DIR = "data"
FIGURES_DIR = "outputs/figures"
EMBEDDINGS_DIR = "outputs/embeddings"

## Clustering (Etapa 1.2)
CLUSTERING_PARAMS = {
    "k_range": list(range(2, 26)),  # K = 2 até 25
    "algorithms": ["kmeans", "kmedoids"],
    "random_state": 42,
    "max_iter": 300
}

## Metodologia para relatório
METHODOLOGY_NOTES = """
1. Dataset: 640 imagens de faces (RecFac), 20 pessoas, ~32 imagens/pessoa
2. Preprocessamento: Grayscale 128x120, normalização [0,1], StandardScaler (z-score)
3. t-SNE: Projeção 2D para visualização da separabilidade
4. PCA: Redução preservando 90%, 80%, 75% da variância
5. UMAP: Redução para 3, 15, 55, 101 dimensões
6. Clustering: K-means e K-medoids com K=2 até 25, avaliação por índice de Dunn
7. Todos os algoritmos com random_state=42 para reprodutibilidade
8. Saídas em CSV para uso posterior e análises comparativas
"""