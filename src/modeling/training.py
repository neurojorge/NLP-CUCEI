# Contenido COMPLETO y CORREGIDO para: src/modeling/training.py
# (V6: Adaptado para Fusión Directa)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

def optimized_train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Entrenamiento V6: Carga el vector de 'features' combinado.
    """
    model.train()
    total_loss = 0
    sentiment_loss_total = 0
    rating_loss_total = 0
    weight_regularization_total = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # --- CARGA SIMPLIFICADA V6 ---
        features = batch['features'].to(device, non_blocking=True)
        dept_weights = batch['dept_weight_loss'].to(device, non_blocking=True) # Peso para la pérdida/escalado
        sentiment_labels = batch['sentiment_label'].to(device, non_blocking=True)
        normalized_ratings = batch['normalized_rating'].to(device, non_blocking=True)
        # --- FIN DE CARGA V6 ---
        
        # Pasamos solo los dos inputs que el modelo V6 necesita
        outputs = model(features, dept_weights)
        
        sentiment_loss = nn.CrossEntropyLoss()(outputs['sentiment_logits'], sentiment_labels)
        rating_loss = nn.MSELoss()(outputs['final_rating'], normalized_ratings)
        weight_reg = torch.mean(torch.abs(outputs['total_weights'] - 0.5)) * 0.01
        
        # Balance dinámico de pérdidas
        if epoch < 5:
            rating_weight = 0.8
            sentiment_weight = 0.2
        elif epoch < 10:
            rating_weight = 0.6
            sentiment_weight = 0.4
        else:
            rating_weight = 0.7
            sentiment_weight = 0.3
        
        total_batch_loss = (rating_weight * rating_loss + 
                           sentiment_weight * sentiment_loss + 
                           weight_reg)
        
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        sentiment_loss_total += sentiment_loss.item()
        rating_loss_total += rating_loss.item()
        weight_regularization_total += weight_reg.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"    📊 Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Total: {total_batch_loss.item():.4f} | "
                  f"Rating: {rating_loss.item():.4f} | "
                  f"Sentiment: {sentiment_loss.item():.4f}")
    
    return {
        'total_loss': total_loss / num_batches,
        'rating_loss': rating_loss_total / num_batches,
        'sentiment_loss': sentiment_loss_total / num_batches,
        'weight_reg': weight_regularization_total / num_batches
    }

def optimized_validate(model, val_loader, device):
    """
    Validación V6: Carga el vector de 'features' combinado.
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    sentiment_preds = []
    sentiment_actuals = []
    weight_analysis = []
    
    with torch.no_grad():
        for batch in val_loader:
            # --- CARGA SIMPLIFICADA V6 ---
            features = batch['features'].to(device, non_blocking=True)
            dept_weights = batch['dept_weight_loss'].to(device, non_blocking=True)
            sentiment_labels = batch['sentiment_label'].to(device, non_blocking=True)
            normalized_ratings = batch['normalized_rating'].to(device, non_blocking=True)
            # --- FIN DE CARGA V6 ---
            
            outputs = model(features, dept_weights)
            
            rating_loss = nn.MSELoss()(outputs['final_rating'], normalized_ratings)
            sentiment_loss = nn.CrossEntropyLoss()(outputs['sentiment_logits'], sentiment_labels)
            batch_loss = 0.7 * rating_loss + 0.3 * sentiment_loss
            
            total_loss += batch_loss.item()
            
            predictions.extend(outputs['final_rating'].cpu().numpy())
            actuals.extend(normalized_ratings.cpu().numpy())
            
            sentiment_pred = torch.softmax(outputs['sentiment_logits'], dim=1)
            sentiment_preds.extend(torch.argmax(sentiment_pred, dim=1).cpu().numpy())
            sentiment_actuals.extend(sentiment_labels.cpu().numpy())
            
            weight_analysis.extend(outputs['total_weights'].cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    weight_analysis = np.array(weight_analysis).flatten()
    
    mse = mean_squared_error(actuals, predictions)
    correlation = np.corrcoef(actuals, predictions)[0,1] if len(set(predictions)) > 1 else 0.0
    sentiment_accuracy = np.mean(np.array(sentiment_preds) == np.array(sentiment_actuals))
    
    return {
        'val_loss': total_loss / len(val_loader),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'correlation': correlation,
        'sentiment_accuracy': sentiment_accuracy,
        'predictions': predictions,
        'actuals': actuals,
        'avg_weight': np.mean(weight_analysis),
        'weight_std': np.std(weight_analysis)
    }