import numpy as np
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import GridSearchCV

def lightweight_optimization(trainer, X_train, y_train, run_optimization=False):
    """
    Optimización ligera de hiperparámetros del mejor modelo
    Solo se ejecuta si run_optimization=True
    """
    print(f"\n🔧 OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 50)
    
    best_model_name = trainer.results.loc[trainer.results["f1_weighted"].idxmax(), "model_name"]
    
    if not run_optimization:
        print(f"⚠️ Optimización OMITIDA por rendimiento")
        print(f"   Usando modelo base: {best_model_name}")
        print(f"   Para activar optimización, cambiar ENABLE_OPTIMIZATION=True")
        return trainer.best_model, {}
    
    print(f"Optimizando: {best_model_name}")
    
    # Grids reducidos para mejor rendimiento
    lightweight_grids = {
        "Random_Forest": {
            "n_estimators": [50, 100],
            "max_depth": [10, None],
            "min_samples_split": [2, 5]
        },
        "Logistic_Regression": {
            "C": [0.1, 1, 10],
            "solver": ['liblinear']
        },
        "Gradient_Boosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.2]
        }
    }
    
    if best_model_name not in lightweight_grids:
        print(f"No hay grid ligero para {best_model_name}")
        return trainer.best_model, {}
    
    # Grid Search reducido
    base_model = trainer.models[best_model_name]
    param_grid = lightweight_grids[best_model_name]
    
    cv_folds = min(3, len(np.unique(y_train)))  # Menos folds
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0  # Sin verbose para reducir output
    )
    
    print(f"Ejecutando Grid Search ligero...")
    start_time = datetime.now()
    
    grid_search.fit(X_train, y_train)
    
    optimization_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Optimización completada en {optimization_time:.1f} segundos")
    print(f"Mejor score: {grid_search.best_score_:.4f}")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_