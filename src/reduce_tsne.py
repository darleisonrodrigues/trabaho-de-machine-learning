"""
t-SNE reduction module.
Responsável por rodar t-SNE em X (após PCA opcional para 50 componentes),
gerar projeção 2D, salvar em CSV e PNG.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from tqdm import tqdm


def apply_tsne(X, perplexity=30, max_iter=1000, random_state=42, 
               use_pca_first=True, pca_components=50):
    """
    Aplica t-SNE para redução de dimensionalidade para 2D (conforme enunciado).
    
    Args:
        X (np.array): Dados de entrada
        perplexity (float): Parâmetro perplexity do t-SNE
        max_iter (int): Número máximo de iterações
        random_state (int): Seed para reprodutibilidade
        use_pca_first (bool): Se deve aplicar PCA primeiro
        pca_components (int): Número de componentes PCA
    
    Returns:
        tuple: (embedding_2d, pca_model ou None)
    """
    print(f"Aplicando t-SNE em dados: {X.shape}")
    
    X_input = X.copy()
    pca_model = None
    
    # Aplicar PCA primeiro se solicitado (recomendado para alta dimensionalidade)
    if use_pca_first and X.shape[1] > pca_components:
        print(f"Aplicando PCA primeiro para {pca_components} componentes...")
        pca_model = PCA(n_components=pca_components, random_state=random_state)
        X_input = pca_model.fit_transform(X)
        print(f"Dados após PCA: {X_input.shape}")
        print(f"Variância explicada pelo PCA: {pca_model.explained_variance_ratio_.sum():.4f}")
    
    # Aplicar t-SNE SEMPRE em 2D (conforme enunciado)
    print(f"Executando t-SNE para 2D com perplexity={perplexity}, max_iter={max_iter}...")
    tsne = TSNE(
        n_components=2,  # SEMPRE 2D conforme enunciado
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1
    )
    
    embedding = tsne.fit_transform(X_input)
    
    print(f"t-SNE concluído. Shape final: {embedding.shape}")
    print(f"KL divergence final: {tsne.kl_divergence_:.4f}")
    
    return embedding, pca_model


def save_tsne_results(embedding, image_paths, output_dir="outputs/embeddings", 
                     filename_prefix="tsne_2d"):
    """
    Salva resultados do t-SNE 2D em CSV (conforme enunciado).
    
    Args:
        embedding (np.array): Resultado do t-SNE 2D
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
        filename_prefix (str): Prefixo do arquivo
    
    Returns:
        str: Caminho do arquivo salvo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar DataFrame para t-SNE 2D
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'image_path': image_paths
    })
    
    # Adicionar informações extras
    df['sample_id'] = range(len(df))
    df['folder'] = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df['filename'] = df['image_path'].apply(lambda x: os.path.basename(x))
    
    # Salvar CSV
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Embedding t-SNE 2D salvo em: {csv_path}")
    
    return csv_path


def create_tsne_visualization(embedding, image_paths, save_path, title="t-SNE Visualization"):
    """
    Cria visualizações do t-SNE.
    
    Args:
        embedding (np.array): Resultado do t-SNE
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura
        title (str): Título do gráfico
    """
    # Extrair labels das pastas (classes)
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    unique_labels = sorted(list(set(labels)))
    
    # Criar mapeamento de label para cor
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    # Criar figura com matplotlib
    plt.figure(figsize=(12, 8))
    
    if embedding.shape[1] == 2:
        # Plot 2D
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=colors, 
                            cmap='tab20', alpha=0.7, s=50)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    else:
        # Plot 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           c=colors, 
                           cmap='tab20', alpha=0.7, s=50)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
    
    plt.title(title)
    
    # Adicionar colorbar com labels
    if len(unique_labels) <= 20:  # Só mostrar legenda se não houver muitas classes
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classes')
        # Definir ticks do colorbar
        tick_positions = list(range(len(unique_labels)))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(unique_labels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualização t-SNE salva em: {save_path}")


def create_interactive_tsne_plot(embedding, image_paths, save_path, title="t-SNE Interactive"):
    """
    Cria visualização interativa do t-SNE com Plotly.
    
    Args:
        embedding (np.array): Resultado do t-SNE
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura HTML
        title (str): Título do gráfico
    """
    # Preparar dados
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    filenames = [os.path.basename(path) for path in image_paths]
    
    if embedding.shape[1] == 2:
        # Plot 2D interativo
        fig = px.scatter(
            x=embedding[:, 0], 
            y=embedding[:, 1],
            color=labels,
            hover_data={'filename': filenames},
            title=title,
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'}
        )
    else:
        # Plot 3D interativo
        fig = px.scatter_3d(
            x=embedding[:, 0], 
            y=embedding[:, 1], 
            z=embedding[:, 2],
            color=labels,
            hover_data={'filename': filenames},
            title=title,
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'z': 't-SNE Component 3'}
        )
    
    fig.write_html(save_path)
    print(f"Visualização interativa t-SNE salva em: {save_path}")


def run_tsne_analysis(X, image_paths, output_dir="outputs", 
                     perplexity_values=[30], use_pca_first=True):
    """
    Executa análise completa do t-SNE com diferentes parâmetros.
    
    Args:
        X (np.array): Dados de entrada
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
        perplexity_values (list): Valores de perplexity para testar
        use_pca_first (bool): Se deve aplicar PCA primeiro
    
    Returns:
        dict: Resultados dos experimentos
    """
    results = {}
    
    for perplexity in perplexity_values:
        print(f"\n=== Executando t-SNE com perplexity={perplexity} ===")
        
        # Aplicar t-SNE
        embedding, pca_model = apply_tsne(
            X, 
            perplexity=perplexity, 
            use_pca_first=use_pca_first
        )
        
        # Definir nomes dos arquivos
        prefix = f"tsne_perp{perplexity}"
        if use_pca_first:
            prefix += "_with_pca"
        
        # Salvar embedding
        csv_path = save_tsne_results(
            embedding, image_paths, 
            os.path.join(output_dir, "embeddings"), 
            prefix
        )
        
        # Criar visualizações
        static_plot_path = os.path.join(output_dir, "figures", f"{prefix}_plot.png")
        create_tsne_visualization(
            embedding, image_paths, static_plot_path,
            f"t-SNE (perplexity={perplexity})"
        )
        
        interactive_plot_path = os.path.join(output_dir, "figures", f"{prefix}_interactive.html")
        create_interactive_tsne_plot(
            embedding, image_paths, interactive_plot_path,
            f"t-SNE Interactive (perplexity={perplexity})"
        )
        
        # Salvar modelo se usado PCA
        if pca_model is not None:
            pca_path = os.path.join(output_dir, "embeddings", f"{prefix}_pca_model.joblib")
            joblib.dump(pca_model, pca_path)
            print(f"Modelo PCA salvo em: {pca_path}")
        
        results[perplexity] = {
            'embedding': embedding,
            'pca_model': pca_model,
            'csv_path': csv_path,
            'static_plot': static_plot_path,
            'interactive_plot': interactive_plot_path
        }
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    from preprocess import load_preprocessed_data
    
    # Carregar dados preprocessados
    processed_path = "data/processed/preprocessed_data.joblib"
    if os.path.exists(processed_path):
        data = load_preprocessed_data(processed_path)
        X_scaled = data['X_scaled']
        image_paths = data['image_paths']
        
        # Executar análise t-SNE
        print("Executando análise t-SNE...")
        results = run_tsne_analysis(
            X_scaled, 
            image_paths,
            perplexity_values=[5, 30, 50],  # Testar diferentes valores
            use_pca_first=True
        )
        
        print("\nAnálise t-SNE concluída!")
        print(f"Resultados salvos para {len(results)} configurações")
        
    else:
        print("Execute primeiro preprocess.py para preprocessar os dados!")