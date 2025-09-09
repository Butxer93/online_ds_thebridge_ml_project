import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clase para limpieza sistemática de datos
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cleaning_log = []
        self.original_shape = None
        
    def clean_dataset(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Pipeline completo de limpieza con operaciones optimizadas
        
        Args:
            df: DataFrame a limpiar
            inplace: Si True, modifica el DataFrame original
        
        Returns:
            DataFrame limpio
        """
        logger.info("Iniciando limpieza optimizada de datos")
        self.original_shape = df.shape
        
        # Solo hacer copia si no es inplace
        if not inplace:
            df = df.copy()
        
        # Pipeline optimizado
        self._handle_missing_values_vectorized(df, inplace=True)
        self._handle_duplicates_optimized(df, inplace=True)  
        self._convert_data_types_efficient(df, inplace=True)
        self._clean_text_fields_vectorized(df, inplace=True)
        
        logger.info(f"Limpieza completada: {self.original_shape} -> {df.shape}")
        return df
    
    def _handle_missing_values_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> None:
        """
        Manejo vectorizado de valores faltantes - SIN LOOPS
        """
        logger.info("Procesando valores faltantes (vectorizado)")
        
        # Convertir strings vacías a NaN de una vez
        df.replace('', np.nan, inplace=inplace)
        
        # Imputaciones simples vectorizadas
        simple_imputations = {
            'Sub-product': 'Not specified',
            'Sub-issue': 'Not specified', 
            'State': 'Unknown',
            'Consumer disputed?': 'No'
        }
        
        for col, fill_value in simple_imputations.items():
            if col in df.columns:
                df[col].fillna(fill_value, inplace=inplace)
        
        # ZIP code por estado - VECTORIZADO con groupby + transform
        if all(col in df.columns for col in ['ZIP code', 'State']):
            self._impute_zip_vectorized(df, inplace=inplace)
    
    def _impute_zip_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> None:
        """
        Imputación de ZIP code vectorizada usando groupby + transform
        REEMPLAZA el loop costoso de Python
        """
        logger.info("Imputando ZIP codes (vectorizado)")
        
        # Crear máscara de valores faltantes
        zip_missing_mask = df['ZIP code'].isnull()
        
        if not zip_missing_mask.any():
            return
        
        # Calcular moda por estado usando groupby + transform
        # Esto es MUCHO más eficiente que loops con .loc
        state_zip_mode = (df[df['ZIP code'].notna()]
                         .groupby('State')['ZIP code']
                         .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None))
        
        # Mapear modas a estados con ZIP faltante - VECTORIZADO
        zip_to_fill = df.loc[zip_missing_mask, 'State'].map(state_zip_mode)
        
        # Asignar valores imputados de una sola vez
        if inplace:
            df.loc[zip_missing_mask, 'ZIP code'] = zip_to_fill
        else:
            df_result = df.copy()
            df_result.loc[zip_missing_mask, 'ZIP code'] = zip_to_fill
            return df_result
    
    def _handle_duplicates_optimized(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Eliminación optimizada de duplicados
        """
        logger.info("Eliminando duplicados")
        initial_rows = len(df)
        
        # Usar inplace para evitar copias innecesarias
        if inplace:
            # Duplicados exactos
            df.drop_duplicates(inplace=True)
            
            # Duplicados por ID
            if 'Complaint ID' in df.columns:
                df.drop_duplicates(subset=['Complaint ID'], inplace=True)
                
            removed = initial_rows - len(df)
            logger.info(f"Eliminados {removed} registros duplicados")
            return df
        else:
            df_clean = df.drop_duplicates()
            if 'Complaint ID' in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=['Complaint ID'])
            return df_clean
    
    def _convert_data_types_efficient(self, df: pd.DataFrame, inplace: bool = True) -> None:
        """
        Conversión eficiente de tipos usando asignación directa
        """
        logger.info("Convirtiendo tipos de datos (eficiente)")
        
        # Fechas - usar to_datetime vectorizado
        date_columns = ['Date received', 'Date sent to company']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Categóricas - conversión por lotes
        categorical_cols = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 
                           'State', 'Company', 'Company response', 
                           'Timely response?', 'Consumer disputed?']
        
        # Filtrar columnas existentes
        existing_categorical = [col for col in categorical_cols if col in df.columns]
        
        # Convertir todas las categóricas de una vez
        for col in existing_categorical:
            df[col] = df[col].astype('category')
        
        # ZIP code como string - preservar ceros iniciales
        if 'ZIP code' in df.columns:
            df['ZIP code'] = df['ZIP code'].astype(str).str.split('.').str[0]
    
    def _clean_text_fields_vectorized(self, df: pd.DataFrame, inplace: bool = True) -> None:
        """
        Limpieza vectorizada de campos de texto
        """
        logger.info("Limpiando campos de texto (vectorizado)")
        
        text_columns = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company']
        
        for col in text_columns:
            if col in df.columns:
                # Operaciones vectorizadas en cadena
                df[col] = (df[col]
                          .astype(str)
                          .str.strip()
                          .str.title())
    
    def get_cleaning_summary(self) -> Dict:
        """
        Resumen de limpieza con métricas de rendimiento
        """
        return {
            'original_rows': self.original_shape[0] if self.original_shape else 0,
            'original_cols': self.original_shape[1] if self.original_shape else 0,
            'cleaning_steps': len(self.cleaning_log),
            'memory_optimized': True,
            'vectorized_operations': True
        }
    
# Ejemplo de uso optimizado
def optimized_cleaning_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo optimizado
    """
    logger.info("Iniciando pipeline optimizado")
    
    # Limpieza optimizada - trabajar inplace cuando sea posible
    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataset(df_raw, inplace=False)  # Primera copia necesaria
    
    logger.info("Pipeline optimizado completado")
    return df_clean
