import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering optimizado para memoria
    """
    
    def __init__(self, df: pd.DataFrame):
        # Trabajar sobre el DataFrame original, no copia
        self.df = df
        self.new_features = []
    
    def create_temporal_features(self, inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        Features temporales optimizadas - operaciones vectorizadas
        """
        logger.info("Creando features temporales (optimizado)")
        
        if 'Date received' not in self.df.columns:
            return None if not inplace else self.df
        
        # Extraer componentes vectorizados - MÁS RÁPIDO que loops
        date_col = self.df['Date received']
        
        # Todas las extracciones en una sola pasada
        temporal_features = {
            'year_received': date_col.dt.year,
            'month_received': date_col.dt.month, 
            'dayofweek_received': date_col.dt.dayofweek,
            'quarter_received': date_col.dt.quarter,
            'is_weekend': date_col.dt.dayofweek.isin([5, 6]).astype('int8'),  # int8 ahorra memoria
            'is_holiday_season': date_col.dt.month.isin([11, 12]).astype('int8')
        }
        
        # Asignar todas las features de una vez
        if inplace:
            for feature_name, values in temporal_features.items():
                self.df[feature_name] = values
                self._log_feature(feature_name, 'Temporal feature')
        
        # Tiempo de procesamiento si ambas fechas existen
        if 'Date sent to company' in self.df.columns:
            processing_days = (self.df['Date sent to company'] - 
                             self.df['Date received']).dt.days
            
            if inplace:
                self.df['processing_days'] = processing_days
                self.df['same_day_processing'] = (processing_days == 0).astype('int8')
                self._log_feature('processing_days', 'Processing time')
        
        return None if inplace else self.df
    
    def create_aggregated_features_optimized(self, inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        Features agregadas usando operaciones eficientes de pandas
        """
        logger.info("Creando features agregadas (optimizado)")
        
        # Contar quejas por empresa - VECTORIZADO
        if 'Company' in self.df.columns:
            # value_counts es más eficiente que loops
            company_counts = self.df['Company'].value_counts()
            
            if inplace:
                # map es vectorizado - MÁS RÁPIDO que loops con .loc
                self.df['company_complaint_count'] = self.df['Company'].map(company_counts)
                
                # cut es vectorizado para categorización
                self.df['company_size'] = pd.cut(
                    self.df['company_complaint_count'], 
                    bins=[0, 10, 50, 200, float('inf')], 
                    labels=['small', 'medium', 'large', 'enterprise']
                )
                
                self._log_feature('company_features', 'Company aggregated features')
        
        # Mismo patrón para estados
        if 'State' in self.df.columns:
            state_counts = self.df['State'].value_counts()
            if inplace:
                self.df['state_complaint_count'] = self.df['State'].map(state_counts)
                self._log_feature('state_complaint_count', 'State aggregated feature')
        
        return None if inplace else self.df
    
    def create_categorical_features_efficient(self, inplace: bool = True) -> Optional[pd.DataFrame]:
        """
        Features categóricas usando mapeo eficiente
        """
        logger.info("Creando features categóricas (eficiente)")
        
        if 'Product' not in self.df.columns:
            return None if not inplace else self.df
        
        # Mapeo de productos - VECTORIZADO con map
        product_mapping = {
            'debt': ['Debt Collection'],
            'credit': ['Credit Card', 'Credit Reporting', 'Credit Report'], 
            'mortgage': ['Mortgage'],
            'banking': ['Bank Account Or Service', 'Checking Or Savings Account'],
            'loan': ['Consumer Loan', 'Student Loan', 'Payday Loan']
        }
        
        # Crear mapeo inverso eficiente
        product_to_category = {}
        for category, products in product_mapping.items():
            for product in products:
                product_to_category[product] = category
        
        if inplace:
            # map es mucho más eficiente que str.contains en loops
            self.df['product_category'] = (self.df['Product']
                                         .map(product_to_category)
                                         .fillna('other'))
            
            self._log_feature('product_category', 'Product categorization')
        
        # Regiones usando mapeo directo - MÁS EFICIENTE
        if 'State' in self.df.columns:
            regions = {
                'northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
                'midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
                'south': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
                'west': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI']
            }
            
            # Crear mapeo inverso
            state_to_region = {}
            for region, states in regions.items():
                for state in states:
                    state_to_region[state] = region
            
            if inplace:
                self.df['region'] = (self.df['State']
                                   .map(state_to_region)
                                   .fillna('unknown'))
                self._log_feature('region', 'Geographic region')
        
        return None if inplace else self.df
    
    def _log_feature(self, feature_name: str, description: str) -> None:
        """Registro de features creadas"""
        self.new_features.append({
            'feature': feature_name,
            'description': description
        })
        logger.info(f"Feature creada: {feature_name} - {description}")

# Ejemplo de uso optimizado
def optimized_engineering_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo optimizado
    """
    logger.info("Iniciando pipeline optimizado")
    
    # Feature engineering optimizado - trabajar inplace
    engineer = FeatureEngineer(df_raw)
    engineer.create_temporal_features(inplace=True)
    engineer.create_aggregated_features_optimized(inplace=True)
    engineer.create_categorical_features_efficient(inplace=True)
    
    logger.info("Pipeline optimizado completado")
    return df_raw

def prepare_final_dataset(df):
    """Preparación final del dataset para modelado"""
    print("\n" + "="*80)
    print("🎯 PREPARACIÓN FINAL PARA MODELADO")
    print("="*80)
    
    df_final = df.copy()
    
    # Eliminar columnas no necesarias para modelado
    columns_to_drop = ["Complaint ID", "Date received", "Date sent to company"]
    columns_to_drop = [col for col in columns_to_drop if col in df_final.columns]
    
    if columns_to_drop:
        df_final = df_final.drop(columns=columns_to_drop)
        print(f"Columnas eliminadas: {columns_to_drop}")
    
    # Verificar balance de la variable objetivo
    if "Company response" in df_final.columns:
        target_distribution = df_final["Company response"].value_counts()
        print(f"\nDistribución de la variable objetivo:")
        for response, count in target_distribution.items():
            pct = (count / len(df_final)) * 100
            print(f"   • {response}: {count:,} ({pct:.1f}%)")
        
        # Calcular ratio de desbalance
        max_class = target_distribution.max()
        min_class = target_distribution.min()
        imbalance_ratio = max_class / min_class
        print(f"\nRatio de desbalance: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            print("⚠️ Dataset desbalanceado detectado - considerar técnicas de balanceo")
        else:
            print("✅ Dataset relativamente balanceado")
    
    # Resumen final
    print(f"\n📊 DATASET FINAL PARA MODELADO:")
    print(f"   • Dimensiones: {df_final.shape}")
    print(f"   • Variables numéricas: {len(df_final.select_dtypes(include=[np.number]).columns)}")
    print(f"   • Variables categóricas: {len(df_final.select_dtypes(include=['object', 'category']).columns)}")
    print(f"   • Valores faltantes: {df_final.isnull().sum().sum()}")
    
    return df_final