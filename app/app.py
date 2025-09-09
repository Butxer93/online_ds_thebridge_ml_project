import streamlit as st
import sys
import os

# Agregar la ruta del directorio src al path de Python
src_path = os.path.join(os.path.dirname(__file__), '../src')
sys.path.append(src_path)

# Importar directamente la clase
from prediction import RobustComplaintPredictor

# Inicializar el predictor (carga automáticamente los modelos)
@st.cache_resource
def load_predictor():
    """Función para cargar el predictor usando cache de Streamlit"""
    models_path = os.path.join(os.path.dirname(__file__), '../models/')
    return RobustComplaintPredictor(models_path=models_path)

st.title('Respuestas de empresas a quejas')
st.write('Introduce los valores de la queja para predecir la reaación de la empresa:')

product = st.text_input('Product', 'Type here...')
issue = st.text_input('Issue', 'Type here...')
state = st.selectbox('State', ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS', 'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX', 'MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI'])
zip_code = st.number_input('ZIP code', min_value=501, max_value=99950)
company = st.text_input('Company', 'Type here...')

if st.button('Predecir'):
    predictor = load_predictor()
    
    features = [
        {
            "Product" : [product],
            "Issue" : [issue],
            "State" : [state],
            "ZIP code" : [zip_code],
            "Company" : [company]
        }
    ]
    
    result = predictor.predict(features)

    if result['prediction_successful']:
        # Mostrar la predicción principal
        st.success(f"Predicción: {result['predicted_response']}")
        st.info(f"Confianza: {result['confidence']:.3f}")
        
        # Mostrar top 3 probabilidades
        st.subheader("Top 3 probabilidades:")
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        for i, (response, prob) in enumerate(sorted_probs[:3], 1):
            st.write(f"{i}. {response}: {prob:.3f}")
    else:
        st.text(f"❌ Error: {result['error']}")
