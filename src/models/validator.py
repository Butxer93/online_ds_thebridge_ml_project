import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import cross_val_score

def cross_validation_analysis(model, X, y, cv=5):
    """
    Análisis de validación cruzada para evaluar estabilidad
    """
    print(f"\n✅ ANÁLISIS DE VALIDACIÓN CRUZADA")
    print("=" * 50)
    
    # Definir métricas para evaluar
    scoring_metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    
    cv_results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        cv_results[metric] = {
            "scores": scores,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores)
        }
    
    print("Resultados de Validación Cruzada:")
    print("=" * 40)
    
    results_df = pd.DataFrame({
        "Métrica": [metric.replace("_", " ").title() for metric in scoring_metrics],
        "Media": [cv_results[metric]["mean"] for metric in scoring_metrics],
        "Std Dev": [cv_results[metric]["std"] for metric in scoring_metrics],
        "Min": [cv_results[metric]["min"] for metric in scoring_metrics],
        "Max": [cv_results[metric]["max"] for metric in scoring_metrics]
    })
    
    print(results_df.round(4).to_string(index=False))
    
    # Visualización de la estabilidad
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(scoring_metrics):
        scores = cv_results[metric]["scores"]
        x_pos = [i] * len(scores)
        plt.scatter(x_pos, scores, alpha=0.6, s=50)
        plt.errorbar(i, cv_results[metric]["mean"], 
                    yerr=cv_results[metric]["std"],
                    fmt="ro", capsize=10, markersize=8)
    
    plt.xticks(range(len(scoring_metrics)), 
              [metric.replace("_", " ").title() for metric in scoring_metrics])
    plt.ylabel("Score")
    plt.title("Distribución de Scores en Validación Cruzada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Evaluar estabilidad
    stability_threshold = 0.05
    unstable_metrics = []
    
    for metric in scoring_metrics:
        if cv_results[metric]["std"] > stability_threshold:
            unstable_metrics.append(metric)
    
    if unstable_metrics:
        print(f"\n⚠️ Métricas con alta variabilidad: {unstable_metrics}")
        print("   Considerar más datos o regularización")
    else:
        print("\n✅ Modelo estable en validación cruzada")
    
    return cv_results