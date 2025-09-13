# (Modelo V6: Fusión Directa de Características)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re  

try:
    # Importamos las funciones V6
    from src.data_processing.utils import get_optimal_weight_iic, PESOS_DIVISIONES, get_sentiment_score
except ImportError:
    print("Error importando utils (V6). Asegúrate de ejecutar desde la raíz del proyecto.")


class OptimizedProfessorModel(nn.Module):
    """
    Versión 6: Fusión Directa de Características (Arquitectura MLP Simple).
    Acepta UN SOLO vector de entrada (Embedding + 3 características escalares).
    """
    
    def __init__(self, embedding_dim, hidden_dim=256, dropout=0.3):
        super(OptimizedProfessorModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # El input es el embedding (ej: 384) + 3 características (Dept, Div, Sent)
        self.input_dim = embedding_dim + 3 
        
        # Encoder/Fusión V6: Un MLP simple y robusto
        self.fusion_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU()
        )
        
        # Cabezas de predicción
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3) 
        )
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    # --- MÉTODO FORWARD (V6 - Simple) ---
    def forward(self, combined_features, weights_for_loss): 
        
        fused = self.fusion_network(combined_features)
        
        sentiment_logits = self.sentiment_head(fused)
        rating_pred_raw = self.rating_head(fused)
        
        # Ponderación final (escalamos la salida usando el peso de confianza de Dept/Div)
        weighted_rating = rating_pred_raw * weights_for_loss.expand_as(rating_pred_raw)
        final_rating = torch.sigmoid(weighted_rating)
        
        return {
            'sentiment_logits': sentiment_logits,
            'rating_pred': rating_pred_raw,
            'final_rating': final_rating,
            'total_weights': weights_for_loss 
        }


class EnhancedProfesorDataset(Dataset):
    """
    Dataset V6: Concatena Embedding + 3 Features (Dept, Div, Sent)
    en un vector de características único.
    """
    
    def __init__(self, embeddings, ratings, departments, divisions, comments, subjects): # (subjects se ignora)
        print("🧹 Creando dataset optimizado V6 (FUSIÓN DIRECTA)...")
        
        clean_data = []
        for i, rating in enumerate(ratings):
            try:
                if rating is not None and not pd.isna(rating):
                    rating_float = float(rating)
                    if 1 <= rating_float <= 10:
                        clean_data.append({
                            'index': i, 'rating': rating_float, 'embedding': embeddings[i],
                            'department': departments[i] if i < len(departments) else 'No encontrado',
                            'division': divisions[i] if divisions and i < len(divisions) else 'Sin división',
                            'comment': comments[i] if i < len(comments) else '',
                        })
            except (ValueError, TypeError): continue
        
        if len(clean_data) == 0:
             print("ADVERTENCIA: No se encontraron datos limpios.")

        
        print("🎯 Calculando pesos (Dept, Div) y Scores de Sentimiento V6...")
        
        all_features = []
        all_dept_weights = [] # Para la pérdida
        all_ratings = [d['rating'] for d in clean_data]
        all_sentiment_labels = torch.LongTensor([
            2 if r >= 8.0 else 1 if r >= 6.0 else 0 for r in all_ratings
        ])

        # --- BUCLE DE CONSTRUCCIÓN DE FEATURES V6 ---
        for i, data in enumerate(clean_data):
            embedding_tensor = torch.FloatTensor(data['embedding']) 
            
            # 1. Calcular los 3 escalares
            dept_weight_scalar = get_optimal_weight_iic(data['department'], data['division'])
            div_key = str(data['division']).upper().strip() if data['division'] else 'Sin división'
            div_weight_scalar = PESOS_DIVISIONES.get(div_key, 0.1)
            sent_score_scalar = get_sentiment_score(data['comment'])

            # 2. Crear tensores escalares
            dept_tensor = torch.FloatTensor([dept_weight_scalar])
            div_tensor = torch.FloatTensor([div_weight_scalar])
            sent_tensor = torch.FloatTensor([sent_score_scalar])

            # 3. CONCATENAR TODO EN UN VECTOR (Ej: 384 + 1 + 1 + 1  ->  387)
            combined_feature_vector = torch.cat([
                embedding_tensor, dept_tensor, div_tensor, sent_tensor
            ], dim=0)
            
            all_features.append(combined_feature_vector)
            all_dept_weights.append(dept_tensor) # Guardamos el peso de Depto por separado

        # --- CREAR TENSORES FINALES ---
        self.features = torch.stack(all_features)
        self.dept_weights_for_loss = torch.stack(all_dept_weights)
        self.ratings = torch.FloatTensor(all_ratings)
        self.normalized_ratings = ((self.ratings - 1) / 9).unsqueeze(1)
        self.sentiment_labels = all_sentiment_labels
        
        print(f"✅ Dataset optimizado V6 creado:")
        print(f"  • Samples válidos: {len(self.ratings)}")
        print(f"  • Dimensión del Vector de Features: {self.features.shape[1]} (Debe ser Emb_Dim + 3)")
        print(f"  • Score de Sentimiento promedio: {torch.mean(self.features[:, -1]):.3f}")
        
    
    def __len__(self):
        return len(self.ratings)
    
    # --- GETITEM (V6) ---
    def __getitem__(self, idx):
        return {
            'features': self.features[idx], # Input principal
            'dept_weight_loss': self.dept_weights_for_loss[idx], # Para la pérdida/escalado
            'rating': self.ratings[idx],
            'normalized_rating': self.normalized_ratings[idx],
            'sentiment_label': self.sentiment_labels[idx],
        }