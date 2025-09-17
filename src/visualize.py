"""
Visualization module.
Funções genéricas para criar visualizações 2D e 3D de embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import os

# Seaborn é opcional para heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def create_scatter_2d(X, labels=None, title="2D Scatter Plot", 
                     xlabel="Component 1", ylabel="Component 2",
                     save_path=None, figsize=(10, 8), alpha=0.7, s=50):
    """
    Cria gráfico de dispersão 2D.
    
    Args:
        X (np.array): Dados 2D (N, 2)
        labels (list): Labels das amostras para colorir
        title (str): Título do gráfico
        xlabel (str): Label do eixo X
        ylabel (str): Label do eixo Y
        save_path (str): Caminho para salvar figura
        figsize (tuple): Tamanho da figura
        alpha (float): Transparência dos pontos
        s (int): Tamanho dos pontos
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = sorted(list(set(labels)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=[colors[i]], label=label, alpha=alpha, s=s)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=alpha, s=s)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico 2D salvo em: {save_path}")
    
    return fig


def create_scatter_3d(X, labels=None, title="3D Scatter Plot",
                     xlabel="Component 1", ylabel="Component 2", zlabel="Component 3",
                     save_path=None, figsize=(12, 8), alpha=0.7, s=50):
    """
    Cria gráfico de dispersão 3D.
    
    Args:
        X (np.array): Dados 3D (N, 3)
        labels (list): Labels das amostras para colorir
        title (str): Título do gráfico
        xlabel (str): Label do eixo X
        ylabel (str): Label do eixo Y
        zlabel (str): Label do eixo Z
        save_path (str): Caminho para salvar figura
        figsize (tuple): Tamanho da figura
        alpha (float): Transparência dos pontos
        s (int): Tamanho dos pontos
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        unique_labels = sorted(list(set(labels)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      c=[colors[i]], label=label, alpha=alpha, s=s)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=alpha, s=s)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico 3D salvo em: {save_path}")
    
    return fig


def create_interactive_scatter_2d(X, labels=None, hover_text=None, 
                                 title="Interactive 2D Scatter",
                                 xlabel="Component 1", ylabel="Component 2",
                                 save_path=None):
    """
    Cria gráfico de dispersão 2D interativo com Plotly.
    
    Args:
        X (np.array): Dados 2D (N, 2)
        labels (list): Labels das amostras para colorir
        hover_text (list): Texto para mostrar ao passar o mouse
        title (str): Título do gráfico
        xlabel (str): Label do eixo X
        ylabel (str): Label do eixo Y
        save_path (str): Caminho para salvar figura HTML
    
    Returns:
        plotly.graph_objects.Figure: Figura criada
    """
    if labels is not None:
        fig = px.scatter(
            x=X[:, 0], 
            y=X[:, 1],
            color=labels,
            hover_name=hover_text,
            title=title,
            labels={'x': xlabel, 'y': ylabel}
        )
    else:
        fig = go.Figure(data=go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>%{x:.3f}, %{y:.3f}<extra></extra>',
            marker=dict(size=6, opacity=0.7)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel
        )
    
    fig.update_layout(
        width=800,
        height=600,
        hovermode='closest'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Gráfico interativo 2D salvo em: {save_path}")
    
    return fig


def create_interactive_scatter_3d(X, labels=None, hover_text=None,
                                 title="Interactive 3D Scatter",
                                 xlabel="Component 1", ylabel="Component 2", zlabel="Component 3",
                                 save_path=None):
    """
    Cria gráfico de dispersão 3D interativo com Plotly.
    
    Args:
        X (np.array): Dados 3D (N, 3)
        labels (list): Labels das amostras para colorir
        hover_text (list): Texto para mostrar ao passar o mouse
        title (str): Título do gráfico
        xlabel (str): Label do eixo X
        ylabel (str): Label do eixo Y
        zlabel (str): Label do eixo Z
        save_path (str): Caminho para salvar figura HTML
    
    Returns:
        plotly.graph_objects.Figure: Figura criada
    """
    if labels is not None:
        fig = px.scatter_3d(
            x=X[:, 0], 
            y=X[:, 1], 
            z=X[:, 2],
            color=labels,
            hover_name=hover_text,
            title=title,
            labels={'x': xlabel, 'y': ylabel, 'z': zlabel}
        )
    else:
        fig = go.Figure(data=go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>%{x:.3f}, %{y:.3f}, %{z:.3f}<extra></extra>',
            marker=dict(size=4, opacity=0.7)
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel
            )
        )
    
    fig.update_layout(
        width=800,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Gráfico interativo 3D salvo em: {save_path}")
    
    return fig


def create_comparison_plot(embeddings_dict, labels=None, save_path=None, 
                          figsize=(16, 12), suptitle="Embeddings Comparison"):
    """
    Cria gráfico comparativo de múltiplos embeddings.
    
    Args:
        embeddings_dict (dict): Dicionário {nome: embedding_array}
        labels (list): Labels das amostras
        save_path (str): Caminho para salvar figura
        figsize (tuple): Tamanho da figura
        suptitle (str): Título principal
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    n_embeddings = len(embeddings_dict)
    
    # Calcular layout da grade
    n_cols = min(3, n_embeddings)
    n_rows = (n_embeddings + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Garantir que axes seja sempre um array 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = None
    if labels is not None:
        unique_labels = sorted(list(set(labels)))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, (name, embedding) in enumerate(embeddings_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if embedding.shape[1] >= 2:
            if labels is not None:
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[colors[i]], label=label, alpha=0.7, s=30)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=30)
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        else:
            ax.hist(embedding[:, 0], bins=30, alpha=0.7)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Frequency')
        
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    # Remover subplots extras
    for idx in range(n_embeddings, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Adicionar legenda se houver labels
    if labels is not None and len(unique_labels) <= 10:
        handles, labels_legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico comparativo salvo em: {save_path}")
    
    return fig


def create_correlation_heatmap(X, feature_names=None, title="Correlation Heatmap",
                              save_path=None, figsize=(10, 8)):
    """
    Cria heatmap de correlação entre features.
    
    Args:
        X (np.array): Dados (N, features)
        feature_names (list): Nomes das features
        title (str): Título do gráfico
        save_path (str): Caminho para salvar figura
        figsize (tuple): Tamanho da figura
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    # Calcular matriz de correlação
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=figsize)
    
    # Criar heatmap
    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(corr_matrix, 
                    annot=True if X.shape[1] <= 10 else False,
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    xticklabels=feature_names if feature_names else False,
                    yticklabels=feature_names if feature_names else False)
    else:
        # Fallback sem seaborn
        im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        if feature_names and len(feature_names) <= 10:
            plt.xticks(range(len(feature_names)), feature_names, rotation=45)
            plt.yticks(range(len(feature_names)), feature_names)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap de correlação salvo em: {save_path}")
    
    return plt.gcf()


def create_embedding_dashboard(embeddings_dict, image_paths, output_dir="outputs/figures"):
    """
    Cria dashboard completo com todas as visualizações.
    
    Args:
        embeddings_dict (dict): Dicionário com embeddings
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
    
    Returns:
        dict: Caminhos dos arquivos criados
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair labels das pastas
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
    filenames = [os.path.basename(path) for path in image_paths]
    
    created_files = {}
    
    # 1. Gráfico comparativo estático
    comparison_path = os.path.join(output_dir, "embeddings_comparison.png")
    create_comparison_plot(embeddings_dict, labels, comparison_path)
    created_files['comparison_static'] = comparison_path
    
    # 2. Visualizações individuais
    for name, embedding in embeddings_dict.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        
        if embedding.shape[1] >= 2:
            # 2D static
            static_2d_path = os.path.join(output_dir, f"{safe_name}_2d.png")
            create_scatter_2d(embedding[:, :2], labels, f"{name} - 2D View", 
                            save_path=static_2d_path)
            created_files[f'{safe_name}_2d'] = static_2d_path
            
            # 2D interactive
            interactive_2d_path = os.path.join(output_dir, f"{safe_name}_2d_interactive.html")
            create_interactive_scatter_2d(embedding[:, :2], labels, filenames,
                                        f"{name} - Interactive 2D", 
                                        save_path=interactive_2d_path)
            created_files[f'{safe_name}_2d_interactive'] = interactive_2d_path
        
        if embedding.shape[1] >= 3:
            # 3D static
            static_3d_path = os.path.join(output_dir, f"{safe_name}_3d.png")
            create_scatter_3d(embedding[:, :3], labels, f"{name} - 3D View",
                            save_path=static_3d_path)
            created_files[f'{safe_name}_3d'] = static_3d_path
            
            # 3D interactive
            interactive_3d_path = os.path.join(output_dir, f"{safe_name}_3d_interactive.html")
            create_interactive_scatter_3d(embedding[:, :3], labels, filenames,
                                        f"{name} - Interactive 3D",
                                        save_path=interactive_3d_path)
            created_files[f'{safe_name}_3d_interactive'] = interactive_3d_path
    
    print(f"\nDashboard criado com {len(created_files)} arquivos em: {output_dir}")
    
    return created_files


if __name__ == "__main__":
    # Exemplo de uso
    print("Testando módulo de visualização...")
    
    # Gerar dados de exemplo
    np.random.seed(42)
    X_2d = np.random.randn(100, 2)
    X_3d = np.random.randn(100, 3)
    labels = ['Class A'] * 50 + ['Class B'] * 50
    
    # Testar visualizações
    test_dir = "outputs/figures/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # 2D
    create_scatter_2d(X_2d, labels, "Test 2D Scatter", 
                     save_path=os.path.join(test_dir, "test_2d.png"))
    
    # 3D
    create_scatter_3d(X_3d, labels, "Test 3D Scatter",
                     save_path=os.path.join(test_dir, "test_3d.png"))
    
    # Interativo 2D
    create_interactive_scatter_2d(X_2d, labels, [f"Sample {i}" for i in range(100)],
                                 save_path=os.path.join(test_dir, "test_2d_interactive.html"))
    
    # Comparativo
    embeddings_test = {
        "Random 2D": X_2d,
        "Random 3D": X_3d[:, :2]  # Usar apenas 2 componentes para comparação
    }
    
    create_comparison_plot(embeddings_test, labels,
                          save_path=os.path.join(test_dir, "test_comparison.png"))
    
    print("Testes de visualização concluídos!")