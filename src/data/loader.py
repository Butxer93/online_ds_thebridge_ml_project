import pandas as pd 

def load_data(file_path):
    """ 
    Carga el dataset de quejas financieras

    Parameters: 
    ----------
    file_path : str
        Ruta del archivo CSV
    
    Returns: 
    ----------
    pd.DataFrame
        Datafrme con los datos cargados
    """

    try:
        # Intentar cargar el archivo
        df = pd.read_csv(file_path, index_col=0)
        print("✅ Datos cargados correctamente")
        print(f"📊 Dimensiones del Dataset: {df.shape}")
        return df
    except FileNotFoundError: 
        print(f"❌ Error: No se encontró el archivo {file_path}")
        print("💡 Asegúrate de que el archivo esté en la ruta correcta")
        return None
    except Exception as e:
        print(f"❌ Error al cargar los datos: {str(e)}")
        return None 
    
def load_processed_data():
    """
    Carga el dataset de quejas financieras

    Parameters: 
    ----------
    file_path : str
        Ruta del archivo CSV
    
    Returns: 
    ----------
    pd.DataFrame
        Datafrme con los datos cargados
    """
    try:
        df = pd.read_csv("../data/processed/01_raw_data.csv")
        print(f"✅ Datos cargados: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo procesado.")
        print("💡 Asegúrate de que el archivo esté en la ruta correcta")
        return None
    except Exception as e:
        print(f"❌ Error al cargar los datos: {str(e)}")
        return None
    
def load_final_data():
    """
    Carga el dataset final del notebook anterior
    """
    try:
        df = pd.read_csv("../data/processed/02_final_data.csv")
        print(f"✅ Datos finales cargados: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠️ No se encontró el archivo final. Creando datos sintéticos...")
        print("💡 Asegúrate de que el archivo esté en la ruta correcta")
        return None
    except Exception as e:
        print(f"❌ Error al cargar los datos: {str(e)}")
        return None