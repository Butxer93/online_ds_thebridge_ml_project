import numpy as np

# Scikit-learn imports
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Balanceo de clases
from sklearn.utils.class_weight import compute_class_weight
try:
    from imblearn.over_sampling import SMOTE
    IMBALANCED_AVAILABLE = True
except ImportError:
    print("⚠️ imblearn no disponible. Se usarán class_weight para balanceo")
    IMBALANCED_AVAILABLE = False

class DataPreprocessor:
    """
    Clase para preprocesamiento de datos para ML
    """
    
    def __init__(self, target_column="Company response"):
        self.target_column = target_column
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_names = None
        
    def prepare_features_target(self, df):
        """
        Separa features y target
        """
        print("\n🎯 PREPARANDO FEATURES Y TARGET")
        print("-" * 40)
        
        # Separar features y target
        if self.target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{self.target_column}' no encontrada")
            
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features: {X.shape[1]} columnas")
        print(f"Target: {len(self.label_encoder.classes_)} clases")
        print(f"Mapeo de clases:")
        for i, clase in enumerate(self.label_encoder.classes_):
            print(f"   • {clase} → {i}")
        
        return X, y_encoded
    
    def create_preprocessor(self, X):
        """
        Crea el preprocesador para las features
        """
        print("\n🔧 CREANDO PREPROCESADOR")
        print("-" * 40)
        
        # Identificar tipos de columnas
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        print(f"Variables numéricas: {len(numeric_features)}")
        print(f"Variables categóricas: {len(categorical_features)}")
        
        # Pipeline para variables numéricas
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        # Pipeline para variables categóricas
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])
        
        # Combinar preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )
        
        print("✅ Preprocesador creado exitosamente")
        return self.preprocessor
    
    def fit_transform_data(self, X_train, X_test):
        """
        Ajusta el preprocesador y transforma los datos
        """
        print("\n🔄 TRANSFORMANDO DATOS")
        print("-" * 40)
        
        # Ajustar y transformar datos de entrenamiento
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Obtener nombres de features después de la transformación
        numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Nombres para features categóricas después de OneHot
        cat_feature_names = []
        if categorical_features:
            encoder = self.preprocessor.named_transformers_["cat"]["onehot"]
            for i, feature in enumerate(categorical_features):
                categories = encoder.categories_[i][1:]  # Skip first category (dropped)
                for cat in categories:
                    cat_feature_names.append(f"{feature}_{cat}")
        
        self.feature_names = numeric_features + cat_feature_names
        
        print(f"Datos transformados:")
        print(f"   • Train: {X_train_processed.shape}")
        print(f"   • Test: {X_test_processed.shape}")
        print(f"   • Features totales: {len(self.feature_names)}")
        
        return X_train_processed, X_test_processed
    

def handle_class_imbalance(X_train, y_train, method="class_weight"):
    """
    Maneja el desbalance de clases
    """
    print(f"\n🔧 MANEJO DE DESBALANCE DE CLASES")
    print("=" * 50)
    
    # Análizar distribución actual
    unique, counts = np.unique(y_train, return_counts=True)
    print("Distribución actual:")
    for cls, count in zip(unique, counts):
        pct = (count / len(y_train)) * 100
        print(f"   • Clase {cls}: {count:,} ({pct:.1f}%)")
    
    imbalance_ratio = counts.max() / counts.min()
    print(f"Ratio de desbalance: {imbalance_ratio:.2f}")
    
    if method == "class_weight":
        # Calcular pesos de clase
        class_weights = compute_class_weight("balanced", 
                                           classes=unique, 
                                           y=y_train)
        class_weight_dict = dict(zip(unique, class_weights))
        print(f"Pesos de clase calculados: {class_weight_dict}")
        return X_train, y_train, class_weight_dict
    
    elif method == "smote" and IMBALANCED_AVAILABLE:
        # Aplicar SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Datos después de SMOTE:")
        unique_new, counts_new = np.unique(y_train_balanced, return_counts=True)
        for cls, count in zip(unique_new, counts_new):
            pct = (count / len(y_train_balanced)) * 100
            print(f"   • Clase {cls}: {count:,} ({pct:.1f}%)")
        
        return X_train_balanced, y_train_balanced, None
    
    else:
        print("Sin balanceo aplicado")
        return X_train, y_train, None