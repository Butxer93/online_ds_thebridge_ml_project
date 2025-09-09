import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import warnings
import pickle
from datetime import datetime
import zipfile

# Suprimir warnings de sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

class RobustComplaintPredictor:
    """
    Sistema robusto y optimizado de predicción de respuestas de quejas.
    Utiliza operaciones vectorizadas y manejo eficiente de memoria.
    """
    
    def __init__(self, models_path: str = '../models/'):
        """
        Inicializa el predictor cargando todos los artefactos del modelo.
        
        Args:
            models_path: Ruta donde se encuentran los modelos guardados
        """
        self.models_path = models_path
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self._load_model_artifacts()
        
        # Configuración optimizada de valores por defecto y mapeos
        self._setup_optimized_mappings()
        
    def _load_model_artifacts(self):
        """Carga todos los artefactos del modelo entrenado"""
        import os
        
        try:
            # Primero intentamos cargar desde el archivo ZIP
            zip_path = f'{self.models_path}final_model.zip'
            pkl_path = f'{self.models_path}final_model.pkl'
            
            model_loaded = False
            
            # Opción 1: Cargar desde ZIP si existe
            if os.path.exists(zip_path):
                print(f"Cargando modelo desde ZIP: {zip_path}")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    with zf.open("final_model.pkl") as f:
                        self.model = pickle.load(f)
                model_loaded = True
            
            # Opción 2: Cargar desde PKL directo si existe
            elif os.path.exists(pkl_path):
                print(f"Cargando modelo desde PKL: {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    self.model = pickle.load(f)
                model_loaded = True
            
            if not model_loaded:
                raise FileNotFoundError(f"No se encontró ni {zip_path} ni {pkl_path}")
            
            # Cargar preprocessor
            preprocessor_path = f'{self.models_path}preprocessor.pkl'
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"No se encontró: {preprocessor_path}")
            
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Cargar label encoder
            label_encoder_path = f'{self.models_path}label_encoder.pkl'
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError(f"No se encontró: {label_encoder_path}")
                
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✅ Todos los artefactos del modelo cargados correctamente")
            
        except FileNotFoundError as e:
            print(f"❌ Error cargando artefactos: {e}")
            print(f"📁 Verificar que existan los archivos en: {self.models_path}")
            
            # Listar archivos disponibles para debug
            if os.path.exists(self.models_path):
                files = os.listdir(self.models_path)
                print(f"📋 Archivos disponibles en {self.models_path}: {files}")
            else:
                print(f"❌ El directorio {self.models_path} no existe")
            
            raise
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            raise
    
    def _setup_optimized_mappings(self):
        """Establece mapeos optimizados para operaciones vectorizadas"""
        
        # Valores por defecto - solo los esenciales
        self.default_values = {
            'Product': 'Credit Card', 'Sub-product': 'Not Specified',
            'Issue': 'Billing Disputes', 'Sub-issue': 'Not Specified',
            'State': 'CA', 'ZIP code': '90210', 'Company': 'Unknown Company',
            'Timely response?': 'Yes', 'Consumer disputed?': 'No',
            'year_received': 2023, 'processing_days': 2
        }
        
        # Mapeo inverso optimizado para productos - VECTORIZADO
        product_categories = {
            'Debt Collection': 'debt', 'Debt collection': 'debt',
            'Credit Card': 'credit', 'Credit card': 'credit', 'Credit Reporting': 'credit',
            'Credit Report': 'credit', 'Mortgage': 'mortgage', 'Fha Mortgage': 'mortgage',
            'Bank Account Or Service': 'banking', 'Bank account or service': 'banking',
            'Checking Or Savings Account': 'banking', 'Consumer Loan': 'loan',
            'Student Loan': 'loan', 'Payday Loan': 'loan', 'Installment Loan': 'loan'
        }
        self.product_to_category = product_categories
        
        # Mapeo inverso optimizado para regiones - VECTORIZADO  
        state_to_region = {}
        regions = {
            'northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
            'midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'south': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
            'west': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI']
        }
        for region, states in regions.items():
            for state in states:
                state_to_region[state] = region
        self.state_to_region = state_to_region
        
        # Valores típicos por región para state_complaint_count - VECTORIZADO
        self.region_complaint_counts = {
            'west': 2000, 'south': 1500, 'northeast': 1000, 
            'midwest': 800, 'unknown': 500
        }
        
        # Keywords para búsqueda vectorizada
        self.keywords = ['fraud', 'identity', 'payment', 'credit', 'debt', 'loan']
    
    def clean_and_validate_input_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Limpieza VECTORIZADA sin loops - inspirada en cleaner.py
        """
        
        if not inplace:
            df = df.copy()
        
        # Limpieza vectorizada en una sola pasada - SIN LOOPS
        df.replace(['', 'None', 'null', 'NULL', np.nan], None, inplace=True)
        
        # Imputaciones vectorizadas por lotes
        critical_fields = ['Product', 'Issue', 'State', 'Company']
        for field in critical_fields:
            if field not in df.columns:
                df[field] = self.default_values[field]
            else:
                df[field].fillna(self.default_values[field], inplace=True)
        
        # Limpieza de texto vectorizada - TODAS las operaciones en cadena
        text_fields = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company', 'State']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).str.strip().str.title()
            elif field in self.default_values:
                df[field] = self.default_values[field]
        
        return df
    
    def create_temporal_features_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Características temporales VECTORIZADAS - basado en engineer.py optimizado
        """
        
        # Año base
        if 'year_received' not in df.columns:
            df['year_received'] = datetime.now().year
        df['year_received'].fillna(datetime.now().year, inplace=True)
        
        # Todas las características temporales de una vez - VECTORIZADO
        base_month, base_day, base_quarter = 6, 1, 2  # Valores promedio
        
        temporal_features = {
            'month_received': df.get('month_received', base_month),
            'dayofweek_received': df.get('dayofweek_received', base_day), 
            'quarter_received': df.get('quarter_received', base_quarter),
            'is_weekend': df.get('is_weekend', 0),
            'is_holiday_season': df.get('is_holiday_season', 0),
            'processing_days': df.get('processing_days', 2)
        }
        
        # Asignación vectorizada de todas las features
        if inplace:
            for feature_name, values in temporal_features.items():
                df[feature_name] = values
        
        # Característica derivada vectorizada
        df['same_day_processing'] = (df['processing_days'] == 0).astype('int8')
        
        return df if not inplace else None
    
    def create_categorical_features_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Características categóricas VECTORIZADAS usando map() - basado en engineer.py
        """
        
        # Categoría de producto - VECTORIZADO con map()
        df['product_category'] = (df['Product']
                                 .map(self.product_to_category)
                                 .fillna('other'))
        
        # Región geográfica - VECTORIZADO con map() 
        df['region'] = (df['State']
                       .map(self.state_to_region)
                       .fillna('unknown'))
        
        return df if not inplace else None
    
    def create_aggregated_features_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Características agregadas VECTORIZADAS - basado en engineer.py optimizado
        """
        
        # Company complaint count - estimación vectorizada basada en product_category
        product_complaint_mapping = {
            'debt': 75, 'credit': 100, 'mortgage': 60, 
            'banking': 80, 'loan': 65, 'other': 50
        }
        
        if 'company_complaint_count' not in df.columns:
            df['company_complaint_count'] = (df['product_category']
                                           .map(product_complaint_mapping)
                                           .fillna(50))
        
        # Company size - VECTORIZADO con pd.cut
        df['company_size'] = pd.cut(
            df['company_complaint_count'],
            bins=[0, 10, 50, 200, float('inf')],
            labels=['small', 'medium', 'large', 'enterprise']
        )
        
        # State complaint count - VECTORIZADO con map()
        if 'state_complaint_count' not in df.columns:
            df['state_complaint_count'] = (df['region']
                                         .map(self.region_complaint_counts)
                                         .fillna(500))
        
        return df if not inplace else None
    
    def create_text_features_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Características de texto VECTORIZADAS - sin loops
        """
        
        # Longitudes - operaciones vectorizadas
        df['issue_length'] = df['Issue'].astype(str).str.len()
        df['sub-issue_length'] = df['Sub-issue'].astype(str).str.len()
        
        # Keywords - VECTORIZADO con str.contains()
        issue_text = df['Issue'].astype(str).str.lower()
        
        for keyword in self.keywords:
            df[f'has_{keyword}'] = issue_text.str.contains(keyword, case=False, na=False).astype('int8')
        
        return df if not inplace else None
    
    def fill_remaining_fields_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Relleno vectorizado de campos restantes - SIN LOOPS
        """
        
        # Campos requeridos con sus valores por defecto
        required_mapping = {
            'Sub-product': 'Not Specified',
            'Sub-issue': 'Not Specified', 
            'ZIP code': '90210',
            'Timely response?': 'Yes',
            'Consumer disputed?': 'No'
        }
        
        # Asignación vectorizada por lotes
        for field, default_val in required_mapping.items():
            if field not in df.columns:
                df[field] = default_val
            else:
                df[field].fillna(default_val, inplace=True)
        
        return df if not inplace else None
    
    def engineer_features_optimized(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Pipeline OPTIMIZADO de feature engineering - trabajo inplace donde es posible
        """
        
        # Conversión inicial - ÚNICA copia necesaria
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()  # Solo una copia al inicio
        
        # Pipeline vectorizado - TODAS las operaciones inplace
        self.clean_and_validate_input_vectorized(df, inplace=True)
        self.create_temporal_features_vectorized(df, inplace=True) 
        self.create_categorical_features_vectorized(df, inplace=True)
        self.create_aggregated_features_vectorized(df, inplace=True)
        self.create_text_features_vectorized(df, inplace=True)
        self.fill_remaining_fields_vectorized(df, inplace=True)
        
        return df
    
    def predict(self, input_data: Union[Dict, pd.DataFrame], return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Predicción optimizada con manejo eficiente de memoria
        """
        try:
            
            # Feature engineering optimizado - UNA sola copia
            df_processed = self.engineer_features_optimized(input_data)
            
            # Alineación eficiente con características esperadas
            expected_features = set(self.preprocessor.feature_names_in_)
            available_features = set(df_processed.columns)
            missing_features = expected_features - available_features
            
            if missing_features:
                print(f"Características faltantes: {missing_features}")
                # Agregado vectorizado de características faltantes
                for feature in missing_features:
                    df_processed[feature] = 0  # Valor por defecto vectorizado
            
            # Reindexado eficiente - sin copias adicionales
            df_processed = df_processed.reindex(columns=self.preprocessor.feature_names_in_, fill_value=0)
            
            # Preprocesamiento y predicción
            X_processed = self.preprocessor.transform(df_processed)
            prediction = self.model.predict(X_processed)[0]
            probabilities = self.model.predict_proba(X_processed)[0]
            
            # Resultado optimizado
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            result = {
                'predicted_response': predicted_class,
                'confidence': float(max(probabilities)),
                'prediction_successful': True,
                'input_completeness': self._calculate_input_completeness_fast(input_data),
                'memory_optimized': True
            }
            
            if return_probabilities:
                # Construcción eficiente del diccionario de probabilidades
                classes = self.label_encoder.classes_
                result['probabilities'] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(classes, probabilities)
                }
            
            return result
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return {
                'predicted_response': 'Error en predicción',
                'confidence': 0.0,
                'prediction_successful': False,
                'error': str(e),
                'memory_optimized': True
            }
    
    def _calculate_input_completeness_fast(self, input_data: Union[Dict, pd.DataFrame]) -> float:
        """
        Cálculo optimizado de completitud - sin conversiones innecesarias
        """
        if isinstance(input_data, dict):
            provided_fields = sum(1 for v in input_data.values() if v is not None and v != '')
            total_fields = len(self.default_values)
        else:
            provided_fields = input_data.notna().sum().sum()
            total_fields = len(self.default_values) * len(input_data)
        
        return provided_fields / total_fields if total_fields > 0 else 0.0
    
def create_complaint_example_optimized(product=None, issue=None, state=None, company=None, **kwargs):
    """
    Función helper optimizada - construcción directa sin copias
    """
    example = {}
    
    # Construcción eficiente del diccionario
    fields = {'Product': product, 'Issue': issue, 'State': state, 'Company': company}
    example.update({k: v for k, v in fields.items() if v is not None})
    example.update(kwargs)
    
    return example

# Función de benchmark para demostrar mejoras de rendimiento
def benchmark_predictor(predictor, n_predictions=1000):
    """
    Benchmark del sistema optimizado
    """
    import time
    
    # Ejemplos de diferentes complejidades
    examples = [
        {'Product': 'Credit Card', 'Issue': 'Billing disputes'},
        {'Product': 'Mortgage'},
        {},
        {'Product': None, 'Issue': '', 'State': 'Unknown'}
    ]
    
    start_time = time.time()
    
    for i in range(n_predictions):
        example = examples[i % len(examples)]
        predictor.predict(example, return_probabilities=False)
    
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / n_predictions
    predictions_per_sec = n_predictions / (end_time - start_time)
    
    return {
        'total_time': end_time - start_time,
        'avg_time_ms': avg_time_ms,
        'predictions_per_sec': predictions_per_sec
    }