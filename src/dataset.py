"""
Dataset module for loading and preprocessing face images.
Responsável por carregar todas as imagens de data/raw/faces,
converter para grayscale 128x120, normalizar [0,1] e achatar em vetores.
"""

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import joblib


def load_images_from_folder(folder_path, target_size=(128, 120)):
    """
    Carrega todas as imagens de uma pasta e suas subpastas.
    
    Args:
        folder_path (str): Caminho para a pasta com imagens
        target_size (tuple): Tamanho alvo (altura, largura)
    
    Returns:
        tuple: (X_flat, image_paths) onde:
            - X_flat: array (N, 15360) com imagens achatadas
            - image_paths: lista com caminhos das imagens
    """
    images = []
    image_paths = []
    
    # Extensões de imagem suportadas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print(f"Carregando imagens de: {folder_path}")
    
    # Percorrer pasta e subpastas
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file.lower())
            
            if ext in valid_extensions:
                image_paths.append(file_path)
    
    print(f"Encontradas {len(image_paths)} imagens")
    
    # Carregar e processar imagens
    for img_path in tqdm(image_paths, desc="Processando imagens"):
        try:
            # Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erro ao carregar: {img_path}")
                continue
                
            # Converter para grayscale
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img
            
            # Redimensionar para 128x120
            img_resized = cv2.resize(img_gray, (target_size[1], target_size[0]))
            
            # Normalizar para [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Achatar em vetor (128*120 = 15360)
            img_flat = img_normalized.flatten()
            
            images.append(img_flat)
            
        except Exception as e:
            print(f"Erro ao processar {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("Nenhuma imagem foi carregada com sucesso!")
    
    # Converter para array numpy
    X_flat = np.array(images)
    
    print(f"Dataset carregado: {X_flat.shape[0]} imagens de {X_flat.shape[1]} features")
    
    return X_flat, image_paths[:len(images)]


def save_dataset(X, image_paths, save_path):
    """
    Salva o dataset processado em arquivo.
    
    Args:
        X (np.array): Array com dados das imagens
        image_paths (list): Lista com caminhos das imagens
        save_path (str): Caminho para salvar
    """
    dataset = {
        'X': X,
        'image_paths': image_paths,
        'shape_info': {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'image_shape': (128, 120)
        }
    }
    
    joblib.dump(dataset, save_path)
    print(f"Dataset salvo em: {save_path}")


def load_dataset(load_path):
    """
    Carrega dataset salvo anteriormente.
    
    Args:
        load_path (str): Caminho do arquivo salvo
    
    Returns:
        tuple: (X, image_paths)
    """
    dataset = joblib.load(load_path)
    print(f"Dataset carregado de: {load_path}")
    print(f"Shape: {dataset['shape_info']}")
    
    return dataset['X'], dataset['image_paths']


def create_sample_info_df(image_paths):
    """
    Cria DataFrame com informações das amostras.
    
    Args:
        image_paths (list): Lista com caminhos das imagens
    
    Returns:
        pd.DataFrame: DataFrame com info das amostras
    """
    import pandas as pd
    
    sample_info = []
    for i, path in enumerate(image_paths):
        # Extrair nome da pasta (classe/pessoa)
        folder_name = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        
        sample_info.append({
            'sample_id': i,
            'image_path': path,
            'filename': filename,
            'folder': folder_name
        })
    
    return pd.DataFrame(sample_info)


if __name__ == "__main__":
    # Exemplo de uso
    raw_data_path = "data/raw/RecFac"
    processed_data_path = "data/processed/dataset.joblib"
    
    if os.path.exists(processed_data_path):
        print("Carregando dataset salvo...")
        X, image_paths = load_dataset(processed_data_path)
    else:
        print("Processando dataset...")
        X, image_paths = load_images_from_folder(raw_data_path)
        save_dataset(X, image_paths, processed_data_path)
    
    # Criar info das amostras
    import pandas as pd
    sample_df = create_sample_info_df(image_paths)
    sample_df.to_csv("data/processed/sample_info.csv", index=False)
    
    print(f"Dataset final: {X.shape}")
    print(f"Primeiras 5 amostras:\n{sample_df.head()}")