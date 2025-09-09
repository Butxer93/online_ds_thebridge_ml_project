import numpy as np
import matplotlib.pyplot as plt

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

def detailed_model_evaluation(model, X_test, y_test, label_encoder, feature_names=None):
    """
    Evaluación detallada del modelo final
    """
    print(f"\n📊 EVALUACIÓN DETALLADA DEL MODELO FINAL")
    print("=" * 60)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Reporte de clasificación
    print("REPORTE DE CLASIFICACIÓN:")
    print("=" * 40)
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Matriz de confusión
    print("\nMATRIZ DE CONFUSIÓN:")
    print("=" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Evaluación Detallada del Modelo Final", fontsize=16, fontweight="bold")
    
    # 1. Matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=target_names)
    disp.plot(ax=axes[0,0], cmap="Blues")
    axes[0,0].set_title("Matriz de Confusión")
    
    # 2. Feature importance (si disponible)
    if hasattr(model, "feature_importances_") and feature_names:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15
        
        axes[0,1].bar(range(len(indices)), importances[indices])
        axes[0,1].set_title("Top 15 Features Más Importantes")
        axes[0,1].set_xticks(range(len(indices)))
        axes[0,1].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    
    # 3. Curva ROC (para clasificación binaria)
    if len(target_names) == 2 and y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        axes[1,0].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        axes[1,0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[1,0].set_xlabel("Tasa de Falsos Positivos")
        axes[1,0].set_ylabel("Tasa de Verdaderos Positivos")
        axes[1,0].set_title("Curva ROC")
        axes[1,0].legend()
    
    # 4. Distribución de probabilidades predichas
    if y_pred_proba is not None:
        max_proba = np.max(y_pred_proba, axis=1)
        axes[1,1].hist(max_proba, bins=20, alpha=0.7, edgecolor="black")
        axes[1,1].set_xlabel("Probabilidad Máxima Predicha")
        axes[1,1].set_ylabel("Frecuencia")
        axes[1,1].set_title("Distribución de Confianza en Predicciones")
        axes[1,1].axvline(np.mean(max_proba), color="red", linestyle="--", 
                         label=f"Media: {np.mean(max_proba):.3f}")
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Métricas finales
    final_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            if len(target_names) == 2:
                final_metrics["auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                final_metrics["auc"] = roc_auc_score(y_test, y_pred_proba, 
                                                   multi_class="ovr", average="weighted")
        except:
            final_metrics['auc'] = np.nan
    
    print(f"\nMÉTRICAS FINALES:")
    print("=" * 30)
    for metric, value in final_metrics.items():
        print(f"{metric.capitalize().replace('_', ' ')}: {value:.4f}")
    
    return final_metrics