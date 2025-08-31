import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def load_model_artifacts():
    """Carga todos los artefactos del modelo"""
    with open('../models/final_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    with open('../models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    return model, preprocessor, label_encoder

def create_all_features(df):
    """
    Aplica todo el feature engineering necesario para que coincida 
    con el preprocesador entrenado
    """
    df_processed = df.copy()

    # 1. CARACTERISTICAS TEMPORALES
    if 'year_received' in df_processed.columns:
        # Crear caracteristicas temporales derivadas
        df_processed['month_received'] = pd.to_datetime(f"{df_processed['year_received'].iloc[0]}-01-01").month
        df_processed['dayofweek_received'] = 1  # Lunes por defecto
        df_processed['quarter_received'] = 1  # Q1 por defecto
        df_processed['is_weekend'] = 0
        df_processed['is_holiday_season'] = 0

        # Tiempo de procesamiento (usar valor por defecto si no se proporciona)
        if 'processing_days' not in df_processed.columns:
            df_processed['processing_days'] = 2  # Valor por defecto
        df_processed['same_day_processing'] = (df_processed['processing_days'] == 0).astype(int)

    # 2. IMPUTACION DE CAMPOS FALTANTES
    # Sub-product
    if 'Sub-product' not in df_processed.columns:
        df_processed['Sub-product'] = 'Not specified'

    # Sub-issue
    if 'Sub-issue' not in df_processed.columns:
        df_processed['Sub-issue'] = 'Not specified'

    # ZIP code
    if 'ZIP code' not in df_processed.columns:
        df_processed['ZIP code'] = '00000'  # Valor por defecto

    # Consumer disputed
    if 'Consumer disputed?' not in df_processed.columns:
        df_processed['Consumer disputed?'] = 'No'

    # Timely response
    if 'Timely response?' not in df_processed.columns:
        df_processed['Timely response?'] = 'Yes'

    # 3. CARACTERISTICAS CATEGORICAS
    # Categoria de producto
    product_mapping = {
        'debt': ['Debt Collection', 'Debt collection'],
        'credit': ['Credit card', 'Credit Card', 'Credit Reporting', 'Credit Report'],
        'mortgage': ['Mortgage'],
        'banking': ['Bank account or service', 'Bank Account Or Service', 'Checking Or Savings Account'],
        'loan': ['Consumer loan', 'Consumer Loan', 'Student Loan', 'Payday loan', 'Payday Loan']
    }

    df_processed['product_category'] = 'other'
    if 'Product' in df_processed.columns:
        product_value = df_processed['Product'].iloc[0]
        for category, products in product_mapping.items():
            for product in products:
                if product.lower() in product_value.lower():
                    df_processed['product_category'] = category
                    break

    # Region geografica
    regions = {
        'northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'south': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
        'west': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI']
    }

    df_processed['region'] = 'unknown'
    if 'State' in df_processed.columns:
        state_value = df_processed['State'].iloc[0]
        for region, states in regions.items():
            if state_value in states:
                df_processed['region'] = region
                break

    # 4. CARACTERISTICAS AGREGADAS (usar valores promedio/tipicos)
    # Company complaint count - usar valores tipicos basados en el tipo de empresa
    df_processed['company_complaint_count'] = 50  # Valor medio tipico

    # Company size basado en company_complaint_count
    count = df_processed['company_complaint_count'].iloc[0]
    if count <= 10:
        df_processed['company_size'] = 'small'
    elif count <= 50:
        df_processed['company_size'] = 'medium'
    elif count <= 200:
        df_processed['company_size'] = 'large'
    else:
        df_processed['company_size'] = 'enterprise'

    # State complaint count
    df_processed['state_complaint_count'] = 100  # Valor medio t�pico

    # 5. CARACTERISTICAS DE TEXTO
    # Issue length
    if 'Issue' in df_processed.columns:
        df_processed['issue_length'] = len(str(df_processed['Issue'].iloc[0]))
    else:
        df_processed['issue_length'] = 20

    # Sub-issue length
    if 'Sub-issue' in df_processed.columns:
        df_processed['sub-issue_length'] = len(str(df_processed['Sub-issue'].iloc[0]))
    else:
        df_processed['sub-issue_length'] = 10

    # 6. PALABRAS CLAVE EN ISSUES
    keywords = ['fraud', 'identity', 'payment', 'credit', 'debt', 'loan']
    issue_text = str(df_processed.get('Issue', '').iloc[0] if 'Issue' in df_processed.columns else '').lower()

    for keyword in keywords:
        df_processed[f'has_{keyword}'] = int(keyword in issue_text)

    return df_processed

def predict_complaint_response(complaint_data):
    """
    Predice la respuesta de la empresa para una queja
    Ahora incluye feature engineering completo

    Parameters:
    -----------
    complaint_data : dict or pd.DataFrame
        Datos de la queja con las caracteristicas basicas requeridas
        Campos minimos requeridos: Product, Issue, State, Company, year_received

    Returns:
    --------
    dict : Prediccion y probabilidades
    """
    # Cargar artefactos
    model, preprocessor, label_encoder = load_model_artifacts()

    # Convertir a DataFrame si es necesario
    if isinstance(complaint_data, dict):
        df = pd.DataFrame([complaint_data])
    else:
        df = complaint_data.copy()

    # Aplicar feature engineering completo
    df_with_features = create_all_features(df)

    # Verificar que tenemos todas las columnas necesarias
    try:
        # Preprocesar datos
        X_processed = preprocessor.transform(df_with_features)
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        print(f"Columnas disponibles: {list(df_with_features.columns)}")
        print(f"Columnas esperadas por el preprocesador: {preprocessor.feature_names_in_}")
        raise

    # Realizar prediccion
    prediction = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]

    # Convertir prediccion a etiqueta original
    predicted_class = label_encoder.inverse_transform([prediction])[0]

    # Crear diccionario de probabilidades por clase
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        class_name = label_encoder.inverse_transform([i])[0]
        prob_dict[class_name] = float(prob)

    return {
        'predicted_response': predicted_class,
        'confidence': float(max(probabilities)),
        'probabilities': prob_dict,
        'features_used': list(df_with_features.columns)
    }

# Funcion auxiliar para crear ejemplos de prueba validos
def create_complaint_example(product, issue, state, company, year_received=2023, processing_days=None):
    """
    Funcion helper para crear ejemplos de queja con la estructura minima requerida

    Parameters:
    -----------
    product : str
        Tipo de producto (ej: 'Credit card', 'Mortgage', 'Debt collection')
    issue : str  
        Descripcion del problema
    state : str
        Estado (codigo de 2 letras, ej: 'CA', 'TX', 'NY')
    company : str
        Nombre de la empresa
    year_received : int
        Año de recepcion de la queja (default: 2023)
    processing_days : int, optional
        Dias de procesamiento (default: calculado automaticamente)
    """
    example = {
        'Product': product,
        'Issue': issue,
        'State': state,
        'Company': company,
        'year_received': year_received
    }

    if processing_days is not None:
        example['processing_days'] = processing_days

    return example

# Ejemplos de uso:
#if __name__ == "__main__":
#    # Ejemplo 1: Usando la funcion helper
#    complaint_1 = create_complaint_example(
#        product='Credit card',
#        issue='Billing disputes and payment issues',
#        state='CA',
#        company='Big Bank Corp',
#        year_received=2023,
#        processing_days=2
#    )
#    
#    # Ejemplo 2: Definicion directa
#    complaint_2 = {
#        'Product': 'Mortgage',
#        'Issue': 'Application processing delays',
#        'State': 'TX', 
#        'Company': 'Mortgage Company LLC',
#        'year_received': 2023
#    }
#    
#    # Ejemplo 3: Con informacion adicional
#    complaint_3 = {
#        'Product': 'Debt collection',
#        'Issue': 'Continued attempts to collect debt not owed and identity fraud concerns',
#        'State': 'NY',
#        'Company': 'Debt Collectors Inc',
#        'year_received': 2023,
#        'Sub-product': 'Medical',
#        'Sub-issue': 'Debt is not mine',
#        'Consumer disputed?': 'Yes',
#        'processing_days': 5
#    }
#    
#    # Probar predicciones
#    print("Testing complaint predictions...\n")
#    
#    for i, complaint in enumerate([complaint_1, complaint_2, complaint_3], 1):
#        try:
#            result = predict_complaint_response(complaint)
#            print(f"Complaint {i}:")
#            print(f"  Input: {complaint}")
#            print(f"  Predicted response: {result['predicted_response']}")
#            print(f"  Confidence: {result['confidence']:.3f}")
#            print(f"  Top probabilities:")
#            # Mostrar top 3 probabilidades
#            sorted_probs = sorted(result['probabilities'].items(), 
#                                key=lambda x: x[1], reverse=True)
#            for response, prob in sorted_probs[:3]:
#                print(f"    - {response}: {prob:.3f}")
#            print()
#        except Exception as e:
#            print(f"Error processing complaint {i}: {e}\n")
