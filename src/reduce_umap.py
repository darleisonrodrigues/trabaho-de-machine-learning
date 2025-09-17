"""
UMAP reduction module.
Responsável por rodar UMAP para 3, 15, 55 e 101 dimensões,
salvar CSVs e gráficos.
"""

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import os
import joblib
from tqdm import tqdm


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, 
               metric='euclidean', random_state=42):
    """
    Aplica UMAP para redução de dimensionalidade.
    
    Args:
        X (np.array): Dados de entrada
        n_components (int): Número de dimensões de saída
        n_neighbors (int): Número de vizinhos para análise local
        min_dist (float): Distância mínima entre pontos no embedding
        metric (str): Métrica de distância
        random_state (int): Seed para reprodutibilidade
    
    Returns:
        tuple: (embedding, umap_model)
    """
    print(f"Aplicando UMAP em dados: {X.shape}")
    print(f"Parâmetros: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # Criar e ajustar modelo UMAP
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    
    # Aplicar transformação
    embedding = umap_model.fit_transform(X)
    
    print(f"UMAP concluído. Shape final: {embedding.shape}")
    
    return embedding, umap_model


def save_umap_results(embedding, image_paths, n_components, output_dir="outputs/embeddings"):
    """
    Salva resultados do UMAP em CSV com nomes conforme enunciado.
    
    Args:
        embedding (np.array): Resultado do UMAP
        image_paths (list): Caminhos das imagens
        n_components (int): Número de componentes
        output_dir (str): Diretório de saída
    
    Returns:
        str: Caminho do arquivo salvo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar DataFrame
    columns = [f'UMAP{i+1}' for i in range(n_components)]
    df = pd.DataFrame(embedding, columns=columns)
    
    # Adicionar informações das amostras
    df['sample_id'] = range(len(df))
    df['image_path'] = image_paths
    df['folder'] = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    df['filename'] = [os.path.basename(path) for path in image_paths]
    
    # Salvar CSV com nome conforme enunciado (umap_3.csv, umap_15.csv, etc.)
    filename = f"umap_{n_components}.csv"
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    
    print(f"UMAP {n_components}D salvo em: {csv_path}")
    
    return csv_path


def create_umap_2d_plot(embedding, image_paths, save_path, title="UMAP 2D"):
    """
    Cria visualização 2D do UMAP.
    
    Args:
        embedding (np.array): Resultado do UMAP 2D
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura
        title (str): Título do gráfico
    """
    # Extrair labels das pastas
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    unique_labels = sorted(list(set(labels)))
    
    # Criar mapeamento de cores
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=colors, cmap='tab20', alpha=0.7, s=50)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title)
    
    # Adicionar colorbar se há múltiplas classes
    if len(unique_labels) > 1:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classes')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot UMAP 2D salvo em: {save_path}")


def create_umap_3d_plot(embedding, image_paths, save_path, title="UMAP 3D"):
    """
    Cria visualização 3D do UMAP.
    
    Args:
        embedding (np.array): Resultado do UMAP 3D
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura
        title (str): Título do gráfico
    """
    # Extrair labels das pastas
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    unique_labels = sorted(list(set(labels)))
    
    # Criar mapeamento de cores
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                        c=colors, cmap='tab20', alpha=0.7, s=50)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title(title)
    
    # Adicionar colorbar se há múltiplas classes
    if len(unique_labels) > 1:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Classes')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot UMAP 3D salvo em: {save_path}")


def create_interactive_umap_plot(embedding, image_paths, save_path, title="UMAP Interactive"):
    """
    Cria visualização interativa do UMAP com Plotly.
    
    Args:
        embedding (np.array): Resultado do UMAP
        image_paths (list): Caminhos das imagens
        save_path (str): Caminho para salvar figura HTML
        title (str): Título do gráfico
    """
    # Preparar dados
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    filenames = [os.path.basename(path) for path in image_paths]
    
    n_components = embedding.shape[1]
    
    if n_components == 2:
        # Plot 2D interativo
        fig = px.scatter(
            x=embedding[:, 0], 
            y=embedding[:, 1],
            color=labels,
            hover_data={'filename': filenames},
            title=title,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'}
        )
    elif n_components == 3:
        # Plot 3D interativo
        fig = px.scatter_3d(
            x=embedding[:, 0], 
            y=embedding[:, 1], 
            z=embedding[:, 2],
            color=labels,
            hover_data={'filename': filenames},
            title=title,
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'}
        )
    else:
        # Para dimensões maiores, usar scatter matrix dos primeiros 3 componentes
        df_plot = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'UMAP3': embedding[:, 2] if n_components > 2 else embedding[:, 1],
            'label': labels,
            'filename': filenames
        })
        
        fig = px.scatter_matrix(
            df_plot, 
            dimensions=['UMAP1', 'UMAP2', 'UMAP3'],
            color='label',
            title=f"{title} - Scatter Matrix (primeiros 3 componentes)"
        )
    
    fig.write_html(save_path)
    print(f"Visualização interativa UMAP salva em: {save_path}")


def analyze_umap_embeddings(umap_results, save_path):
    """
    Analisa e compara diferentes embeddings UMAP.
    
    Args:
        umap_results (dict): Resultados do UMAP para diferentes dimensões
        save_path (str): Caminho para salvar análise
    """
    analysis = {}
    
    for n_components, result in umap_results.items():
        embedding = result['embedding']
        
        # Calcular estatísticas básicas
        stats = {
            'n_components': n_components,
            'shape': embedding.shape,
            'statistics': {}
        }
        
        for i in range(min(embedding.shape[1], 5)):  # Analisar até 5 componentes
            component = embedding[:, i]
            stats['statistics'][f'UMAP{i+1}'] = {
                'mean': float(component.mean()),
                'std': float(component.std()),
                'min': float(component.min()),
                'max': float(component.max()),
                'median': float(np.median(component))
            }
        
        analysis[f'umap_{n_components}d'] = stats
    
    # Salvar como JSON
    import json
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Análise UMAP salva em: {save_path}")


def create_umap_comparison_plot(umap_results, save_path):
    """
    Cria gráfico comparativo dos UMAPs 2D.
    
    Args:
        umap_results (dict): Resultados do UMAP
        save_path (str): Caminho para salvar figura
    """
    # Filtrar apenas UMAPs 2D e 3D para visualização
    plot_results = {k: v for k, v in umap_results.items() if k <= 3}
    
    n_plots = len(plot_results)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=(15, 5))
    if n_plots == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green']
    
    for idx, (n_comp, result) in enumerate(plot_results.items()):
        if idx >= 3:  # Limitar a 3 plots
            break
            
        embedding = result['embedding']
        
        if n_comp == 2:
            axes[idx].scatter(embedding[:, 0], embedding[:, 1], 
                            alpha=0.6, c=colors[idx], s=30)
            axes[idx].set_xlabel('UMAP 1')
            axes[idx].set_ylabel('UMAP 2')
        elif n_comp == 3:
            axes[idx].scatter(embedding[:, 0], embedding[:, 1], 
                            alpha=0.6, c=colors[idx], s=30)
            axes[idx].set_xlabel('UMAP 1')
            axes[idx].set_ylabel('UMAP 2')
        
        axes[idx].set_title(f'UMAP {n_comp}D')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico comparativo UMAP salvo em: {save_path}")


def run_umap_analysis(X, image_paths, output_dir="outputs", 
                     n_components_list=[3, 15, 55, 101], 
                     n_neighbors=15, min_dist=0.1):
    """
    Executa análise completa do UMAP para diferentes dimensões.
    
    Args:
        X (np.array): Dados de entrada
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
        n_components_list (list): Lista de dimensões para testar
        n_neighbors (int): Número de vizinhos
        min_dist (float): Distância mínima
    
    Returns:
        dict: Resultados completos da análise UMAP
    """
    print("=== Iniciando Análise UMAP ===")
    
    umap_results = {}
    
    for n_components in n_components_list:
        print(f"\n--- UMAP para {n_components} dimensões ---")
        
        # Aplicar UMAP
        embedding, umap_model = apply_umap(
            X, 
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        
        # Salvar embedding
        csv_path = save_umap_results(
            embedding, image_paths, n_components,
            os.path.join(output_dir, "embeddings")
        )
        
        # Salvar modelo
        model_path = os.path.join(output_dir, "embeddings", f"umap_{n_components}d_model.joblib")
        joblib.dump(umap_model, model_path)
        print(f"Modelo UMAP {n_components}D salvo em: {model_path}")
        
        # Criar visualizações
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        if n_components == 2:
            # Plot 2D estático
            static_2d_path = os.path.join(figures_dir, f"umap_{n_components}d_plot.png")
            create_umap_2d_plot(embedding, image_paths, static_2d_path, f"UMAP {n_components}D")
            
        elif n_components == 3:
            # Plot 3D estático
            static_3d_path = os.path.join(figures_dir, f"umap_{n_components}d_plot.png")
            create_umap_3d_plot(embedding, image_paths, static_3d_path, f"UMAP {n_components}D")
        
        # Plot interativo (para 2D, 3D ou scatter matrix)
        interactive_path = os.path.join(figures_dir, f"umap_{n_components}d_interactive.html")
        create_interactive_umap_plot(embedding, image_paths, interactive_path, 
                                   f"UMAP {n_components}D Interactive")
        
        umap_results[n_components] = {
            'embedding': embedding,
            'model': umap_model,
            'csv_path': csv_path,
            'model_path': model_path
        }
    
    # Criar análise comparativa
    analysis_path = os.path.join(output_dir, "embeddings", "umap_analysis.json")
    analyze_umap_embeddings(umap_results, analysis_path)
    
    # Gráfico comparativo
    comparison_path = os.path.join(output_dir, "figures", "umap_comparison.png")
    create_umap_comparison_plot(umap_results, comparison_path)
    
    print("\n=== Análise UMAP Concluída ===")
    print(f"Resultados salvos para {len(n_components_list)} dimensões: {n_components_list}")
    
    return {
        'umap_results': umap_results,
        'analysis': analysis_path,
        'comparison_plot': comparison_path
    }


if __name__ == "__main__":
    # Exemplo de uso
    from preprocess import load_preprocessed_data
    
    # Carregar dados preprocessados
    processed_path = "data/processed/preprocessed_data.joblib"
    if os.path.exists(processed_path):
        data = load_preprocessed_data(processed_path)
        X_scaled = data['X_scaled']
        image_paths = data['image_paths']
        
        # Executar análise UMAP
        print("Executando análise UMAP...")
        results = run_umap_analysis(
            X_scaled, 
            image_paths,
            n_components_list=[2, 3, 15, 55, 101],  # Incluir 2D para visualização
            n_neighbors=15,
            min_dist=0.1
        )
        
        print("\nAnálise UMAP concluída!")
        
    else:
        print("Execute primeiro preprocess.py para preprocessar os dados!")