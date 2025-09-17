"""
Preprocess module for data standardization and scaling.
Responsável por aplicar StandardScaler (z-score), salvar e carregar versões processadas.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


class DataPreprocessor:
    """Classe para preprocessamento de dados com StandardScaler."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X):
        """
        Ajusta o scaler nos dados e transforma.
        
        Args:
            X (np.array): Dados de entrada (N, features)
        
        Returns:
            np.array: Dados padronizados (z-score)
        """
        print(f"Aplicando StandardScaler em dados: {X.shape}")
        
        # Aplicar z-score normalization
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        print(f"Dados padronizados:")
        print(f"  - Média: {X_scaled.mean():.6f}")
        print(f"  - Desvio padrão: {X_scaled.std():.6f}")
        print(f"  - Min: {X_scaled.min():.6f}")
        print(f"  - Max: {X_scaled.max():.6f}")
        
        return X_scaled
    
    def transform(self, X):
        """
        Transforma novos dados usando scaler já ajustado.
        
        Args:
            X (np.array): Dados de entrada
        
        Returns:
            np.array: Dados padronizados
        """
        if not self.is_fitted:
            raise ValueError("Scaler deve ser ajustado primeiro com fit_transform()")
        
        return self.scaler.transform(X)
    
    def save_scaler(self, save_path):
        """
        Salva o scaler ajustado.
        
        Args:
            save_path (str): Caminho para salvar
        """
        if not self.is_fitted:
            raise ValueError("Scaler deve ser ajustado primeiro")
        
        joblib.dump(self.scaler, save_path)
        print(f"Scaler salvo em: {save_path}")
    
    def load_scaler(self, load_path):
        """
        Carrega scaler salvo anteriormente.
        
        Args:
            load_path (str): Caminho do scaler salvo
        """
        self.scaler = joblib.load(load_path)
        self.is_fitted = True
        print(f"Scaler carregado de: {load_path}")


def preprocess_and_save_data(X, image_paths, output_dir="data/processed"):
    """
    Preprocessa dados e salva versões processadas.
    
    Args:
        X (np.array): Dados originais
        image_paths (list): Caminhos das imagens
        output_dir (str): Diretório de saída
    
    Returns:
        tuple: (X_scaled, preprocessor)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar e aplicar preprocessador
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    # Salvar scaler
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    preprocessor.save_scaler(scaler_path)
    
    # Salvar dados preprocessados
    processed_data = {
        'X_scaled': X_scaled,
        'X_original': X,
        'image_paths': image_paths,
        'preprocessing_info': {
            'method': 'StandardScaler (z-score)',
            'original_shape': X.shape,
            'scaled_shape': X_scaled.shape,
            'original_stats': {
                'mean': float(X.mean()),
                'std': float(X.std()),
                'min': float(X.min()),
                'max': float(X.max())
            },
            'scaled_stats': {
                'mean': float(X_scaled.mean()),
                'std': float(X_scaled.std()),
                'min': float(X_scaled.min()),
                'max': float(X_scaled.max())
            }
        }
    }
    
    processed_path = os.path.join(output_dir, "preprocessed_data.joblib")
    joblib.dump(processed_data, processed_path)
    print(f"Dados preprocessados salvos em: {processed_path}")
    
    return X_scaled, preprocessor


def load_preprocessed_data(processed_path):
    """
    Carrega dados preprocessados salvos.
    
    Args:
        processed_path (str): Caminho dos dados preprocessados
    
    Returns:
        dict: Dados carregados
    """
    data = joblib.load(processed_path)
    print(f"Dados preprocessados carregados de: {processed_path}")
    print(f"Info do preprocessamento:")
    for key, value in data['preprocessing_info'].items():
        print(f"  {key}: {value}")
    
    return data


def create_preprocessing_report(X_original, X_scaled, save_path):
    """
    Cria relatório do preprocessamento.
    
    Args:
        X_original (np.array): Dados originais
        X_scaled (np.array): Dados padronizados
        save_path (str): Caminho para salvar relatório
    """
    import matplotlib.pyplot as plt
    
    # Calcular estatísticas
    stats = {
        'Original': {
            'mean': X_original.mean(),
            'std': X_original.std(),
            'min': X_original.min(),
            'max': X_original.max(),
            'median': np.median(X_original)
        },
        'Scaled': {
            'mean': X_scaled.mean(),
            'std': X_scaled.std(),
            'min': X_scaled.min(),
            'max': X_scaled.max(),
            'median': np.median(X_scaled)
        }
    }
    
    # Criar gráfico comparativo
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogramas
    ax1.hist(X_original.flatten(), bins=50, alpha=0.7, color='blue')
    ax1.set_title('Distribuição - Dados Originais')
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Frequência')
    
    ax2.hist(X_scaled.flatten(), bins=50, alpha=0.7, color='red')
    ax2.set_title('Distribuição - Dados Padronizados')
    ax2.set_xlabel('Valor (z-score)')
    ax2.set_ylabel('Frequência')
    
    # Box plots
    ax3.boxplot([X_original.flatten()], labels=['Original'])
    ax3.set_title('Box Plot - Dados Originais')
    ax3.set_ylabel('Valor')
    
    ax4.boxplot([X_scaled.flatten()], labels=['Scaled'])
    ax4.set_title('Box Plot - Dados Padronizados')
    ax4.set_ylabel('Valor (z-score)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Relatório de preprocessamento salvo em: {save_path}")
    
    # Salvar estatísticas em CSV
    stats_df = pd.DataFrame(stats).T
    stats_csv_path = save_path.replace('.png', '_stats.csv')
    stats_df.to_csv(stats_csv_path)
    print(f"Estatísticas salvas em: {stats_csv_path}")


if __name__ == "__main__":
    # Exemplo de uso
    from dataset import load_dataset
    
    # Carregar dados
    dataset_path = "data/processed/dataset.joblib"
    if os.path.exists(dataset_path):
        X, image_paths = load_dataset(dataset_path)
        
        # Preprocessar dados
        X_scaled, preprocessor = preprocess_and_save_data(X, image_paths)
        
        # Criar relatório
        report_path = "outputs/figures/preprocessing_report.png"
        create_preprocessing_report(X, X_scaled, report_path)
        
        print(f"Preprocessamento concluído!")
        print(f"Shape original: {X.shape}")
        print(f"Shape padronizada: {X_scaled.shape}")
    else:
        print("Execute primeiro o dataset.py para carregar os dados!")