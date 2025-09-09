import streamlit as st
import sys
import os

# Obtener la ruta base del proyecto de forma más robusta
if 'mount' in os.getcwd():  # Streamlit Cloud
    # En Streamlit Cloud, usar la ruta base del proyecto
    project_root = '/mount/src/online_ds_thebridge_ml_project'
else:  # Local
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Agregar la ruta del directorio src al path de Python
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

print(f"🔍 Proyecto base: {project_root}")
print(f"🔍 Ruta src: {src_path}")

# Importar directamente la clase
try:
    from prediction import RobustComplaintPredictor
    print("✅ Clase RobustComplaintPredictor importada correctamente")
except ImportError as e:
    print(f"❌ Error importando RobustComplaintPredictor: {e}")
    st.error(f"Error importando el predictor: {e}")
    st.stop()

# Inicializar el predictor (carga automáticamente los modelos)
@st.cache_resource
def load_predictor():
    """Función para cargar el predictor usando cache de Streamlit"""
    
    # Los archivos están en la raíz del proyecto, no en models/
    models_path = project_root  # Usar la raíz directamente
    
    print(f"🔍 Buscando modelos en: {models_path}")
    print(f"🔍 Directorio existe: {os.path.exists(models_path)}")
    
    if os.path.exists(models_path):
        # Verificar archivos específicos del modelo
        required_files = ['final_model.zip', 'preprocessor.pkl', 'label_encoder.pkl']
        for file in required_files:
            file_path = os.path.join(models_path, file)
            print(f"🔍 {file}: {'✅' if os.path.exists(file_path) else '❌'}")
    
    # Asegurar que la ruta termine con /
    if not models_path.endswith('/'):
        models_path += '/'
    
    return RobustComplaintPredictor(models_path=models_path)

st.title('Respuestas de empresas a quejas')
st.write('Introduce los valores de la queja para predecir la reacción de la empresa:')

product = st.text_input('Product', 'Credit Card')
issue = st.text_input('Issue', 'Billing disputes')
state = st.selectbox('State', ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS', 'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX', 'MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'CA', 'OR', 'WA', 'AK', 'HI'])
zip_code = st.number_input('ZIP code', min_value=501, max_value=99950, value=90210)
company = st.text_input('Company', 'Bank Example')

if st.button('Predecir'):
    try:
        predictor = load_predictor()
        
        features = {
            "Product": product,
            "Issue": issue,
            "State": state,
            "ZIP code": zip_code,
            "Company": company
        }
        
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
            st.error(f"❌ Error: {result['error']}")
            
    except Exception as e:
        st.error(f"❌ Error general: {str(e)}")
        # Debug adicional
        st.write("**Información de debug:**")
        st.write(f"- Directorio actual: {os.getcwd()}")
        st.write(f"- Proyecto base detectado: {project_root}")
        st.write(f"- Ruta src: {src_path}")
        
        # Mostrar estructura disponible
        if os.path.exists(project_root):
            st.write("**Estructura del proyecto:**")
            for item in os.listdir(project_root):
                item_path = os.path.join(project_root, item)
                if os.path.isdir(item_path):
                    st.write(f"📁 {item}/")
                    if item == 'models':
                        try:
                            models_files = os.listdir(item_path)
                            for file in models_files:
                                st.write(f"   📄 {file}")
                        except:
                            st.write("   ❌ No se puede leer contenido")
                else:
                    st.write(f"📄 {item}")