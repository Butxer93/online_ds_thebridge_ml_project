import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import learning_curve

def simplified_interpretability_analysis(model, X_test, feature_names, label_encoder):
    """
    Análisis de interpretabilidad simplificado para mejor rendimiento
    """
    print(f"\nPENDIENTE ANALISIS DE INTERPRETABILIDAD")
    print("=" * 50)
    
    # 1. Feature Importance (solo para modelos tree-based)
    if hasattr(model, "feature_importances_"):
        print("IMPORTANCIA DE CARACTERISTICAS:")
        print("-" * 40)
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        print("Top 10 características más importantes:")
        for _, row in feature_importance_df.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
        
        # Visualización simple
        plt.figure(figsize=(12, 6))
        top_features = feature_importance_df.head(10)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importancia")
        plt.title("Top 10 Características Más Importantes")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("Modelo sin feature importance disponible")
    
    # 2. Análisis básico de predicciones
    print(f"\nANALISIS BASICO DE PREDICCIONES:")
    print("-" * 40)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_pred == i
        class_count = np.sum(class_mask)
        
        if class_count > 0 and y_pred_proba is not None:
            avg_confidence = np.mean(y_pred_proba[class_mask, i])
            print(f"   • {class_name}: {class_count} predicciones (confianza promedio: {avg_confidence:.3f})")
    
    print("\nNOTA: Análisis SHAP omitido por rendimiento")
    print("      Para análisis completo, instalar SHAP y activar en código")

def create_simple_learning_curves(model, X_train, y_train, cv=3):
    """
    Curvas de aprendizaje simplificadas
    """
    print(f"\nCREANDO CURVAS DE APRENDIZAJE SIMPLIFICADAS")
    print("-" * 40)
    
    # Menos puntos para mejor rendimiento
    train_sizes = np.linspace(0.2, 1.0, 5)
    
    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train, 
            train_sizes=train_sizes,
            cv=cv, 
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42
        )
        
        # Calcular medias y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plotear curvas de aprendizaje
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Entrenamiento")
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, "o-", color="red", label="Validación")
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel("Número de Muestras de Entrenamiento")
        plt.ylabel("F1-Score Weighted")
        plt.title("Curvas de Aprendizaje")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Análisis de overfitting
        final_train_score = train_mean[-1]
        final_val_score = val_mean[-1]
        overfitting_gap = final_train_score - final_val_score
        
        print(f"Score final entrenamiento: {final_train_score:.4f}")
        print(f"Score final validación: {final_val_score:.4f}")
        print(f"Gap (overfitting): {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            print("ADVERTENCIA: Posible overfitting detectado")
        else:
            print("OK: Overfitting controlado")
            
    except Exception as e:
        print(f"Error en curvas de aprendizaje: {str(e)}")