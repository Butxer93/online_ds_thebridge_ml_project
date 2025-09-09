import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Modelos
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

class ModelTrainer:
    """
    Clase para entrenar y evaluar múltiples modelos
    """
    
    def __init__(self, class_weights=None):
        self.class_weights = class_weights
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def define_models(self):
        """
        Define los modelos a entrenar
        """
        print("\n🤖 DEFINIENDO MODELOS DE ML")
        print("=" * 50)
        
        models = {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            
            "Logistic_Regression": LogisticRegression(
                random_state=42,
                class_weight="balanced" if self.class_weights is None else None,
                max_iter=1000
            ),
            
            "Random_Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced" if self.class_weights is None else None,
                n_jobs=-1
            ),
            
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            ),
            
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            
            "Decision_Tree": DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced" if self.class_weights is None else None
            )
        }
        
        self.models = models
        print(f"Modelos definidos: {len(models)}")
        for name in models.keys():
            print(f"   • {name}")
        
        return models
    
    def train_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Entrena y evalúa todos los modelos
        """
        print(f"\n⌛ ENTRENANDO Y EVALUANDO MODELOS")
        print("=" * 50)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            start_time = datetime.now()
            
            # Entrenar modelo
            if self.class_weights is not None and hasattr(model, "class_weight"):
                model.set_params(class_weight=self.class_weights)
            
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calcular métricas
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics["model_name"] = name
            metrics["training_time"] = (datetime.now() - start_time).total_seconds()
            
            results.append(metrics)
            
            print(f"   ✅ {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calcula métricas de evaluación
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        }
        
        # AUC para clasificación multiclase
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics["auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["auc"] = roc_auc_score(y_true, y_pred_proba, 
                                                 multi_class="ovr", average="weighted")
            except:
                metrics["auc"] = np.nan
        else:
            metrics["auc"] = np.nan
            
        return metrics
    
    def get_model_comparison(self):
        """
        Compara los resultados de todos los modelos
        """
        if self.results is None or len(self.results) == 0:
            print("No hay resultados disponibles")
            return None
            
        print(f"\n📊 COMPARACIÓN DE MODELOS")
        print("=" * 80)
        
        # Ordenar por F1-score weighted
        comparison = self.results.sort_values("f1_weighted", ascending=False)
        
        # Mostrar tabla de resultados
        display_cols = ["model_name", "accuracy", "f1_weighted", "precision_weighted", 
                       "recall_weighted", "auc", "training_time"]
        
        print(comparison[display_cols].round(4).to_string(index=False))
        
        # Identificar mejor modelo
        best_idx = comparison["f1_weighted"].idxmax()
        best_model_name = comparison.loc[best_idx, "model_name"]
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"   • F1-Score: {comparison.loc[best_idx, 'f1_weighted']:.4f}")
        print(f"   • Accuracy: {comparison.loc[best_idx, 'accuracy']:.4f}")
        print(f"   • AUC: {comparison.loc[best_idx, 'auc']:.4f}")
        
        return comparison
    
    def plot_model_comparison(self):
        """
        Visualiza la comparación de modelos
        """
        if self.results is None or len(self.results) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comparación de Modelos de ML", fontsize=16, fontweight="bold")
        
        # 1. Accuracy
        results_sorted = self.results.sort_values('accuracy')
        axes[0,0].barh(results_sorted['model_name'], results_sorted["accuracy"])
        axes[0,0].set_title('Accuracy por Modelo')
        axes[0,0].set_xlabel('Accuracy')
        
        # 2. F1-Score
        results_sorted = self.results.sort_values("f1_weighted")
        axes[0,1].barh(results_sorted["model_name"], results_sorted["f1_weighted"])
        axes[0,1].set_title("F1-Score Weighted por Modelo")
        axes[0,1].set_xlabel("F1-Score")
        
        # 3. AUC (si disponible)
        if not self.results["auc"].isna().all():
            results_sorted = self.results.sort_values("auc", na_position="first")
            axes[1,0].barh(results_sorted["model_name"], results_sorted["auc"])
            axes[1,0].set_title("AUC por Modelo")
            axes[1,0].set_xlabel("AUC")
        
        # 4. Tiempo de entrenamiento
        results_sorted = self.results.sort_values("training_time")
        axes[1,1].barh(results_sorted["model_name"], results_sorted["training_time"])
        axes[1,1].set_title("Tiempo de Entrenamiento (segundos)")
        axes[1,1].set_xlabel("Tiempo (s)")
        
        plt.tight_layout()
        plt.show()