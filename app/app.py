import streamlit as st
import sys

sys.path.append("../src")
from prediction import RobustComplaintPredictor

st.title('Respuestas de empresas a quejas')
st.write('Introduce los valores de la queja para predecir la reaación de la empresa:')

product = st.text_input('Product', 'Type here...')
issue = st.text_input('Issue', 'Type here...')
state = st.selectbox('State', ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS', 'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX', 'MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI'])
zip_code = st.number_input('ZIP code', min_value=501, max_value=99950)
company = st.text_input('Company', 'Type here...')

if st.button('Predecir'):
    predictor = RobustComplaintPredictor()
    
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
        # Mostrar top 3 probabilidades
        st.text(f"  Top probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        for response, prob in sorted_probs[:3]:
            print(f"    - {response}: {prob:.3f}")
    else:
        st.text(f"❌ Error: {result['error']}")
