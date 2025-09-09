import pandas as pd 

def explore_data_structure(df):
    """ 
    Explora la estructura básica del dataset

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame a explorar
    """

    print("\n" + "="*80)
    print("📋 EXPLORACIÓN DE LA ESTRUCTURA DE DATOS")
    print("="*80)

    # Información básica
    print(f"\n📊 DIMENSIONES:")
    print(f"   · Filas: {df.shape[0]:,}")
    print(f"   · Columnas: {df.shape[1]:,}")

    # Tipos de datos
    print(f"\n💿 TIPOS DE DATOS:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   · {dtype}: {count} columnas")

    # Información detallada por columna
    print(f"\n📝 INFORMACIÓN POR COLUMNA:")
    info_data = []
    for col in df.columns:
        info_data.append({
            "Columna": col,
            "Tipo": str(df[col].dtype),
            "No nulos": df[col].count(),
            "% Nulos": round((df[col].isnull().sum() / len(df)) * 100, 2),
            "Valores únicos": df[col].nunique()
        })

    info_df = pd.DataFrame(info_data)
    print(info_df.to_string(index=False))

    return info_df

def analyze_data_quality(df):
    """ 
    Analiza la calidad de los datos

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    """

    print("\n" + "="*80)
    print("🔍 ANÁLISIS DE LA CALIDAD DE DATOS")
    print("="*80)

    # Valores faltantes
    ''' 
    El método más básico sería:
    print(df.isnull().sum())
    '''
    print("\n❓ VALORES FALTANTES:") 
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Columna": missing_data.index,
        "Valores Faltantes": missing_data.values,
        "Porcentaje": missing_percentage.values
    }). sort_values("Porcentaje", ascending=False)

    print(missing_summary[missing_summary["Porcentaje"] > 0].to_string(index=False))

    # Duplicados
    duplicates = df.duplicated().sum()
    print(f"\n🔄 REGISTROS DUPLICADOS:")
    print(f"   · Total: {duplicates:,}")
    print(f"   · Porcentaje: {(duplicates/len(df)*100):.2f}%")

    # Valores únicos en columnas categóricas
    print(f"\n💿 CARDINALIDAD DE VARIABLES CATEGÓRICAS:")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"   · {col}: {unique_count:,} valores únicos")

    return missing_summary 

def explore_key_variables(df):
    """
    Explora las variables más importantes del dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a explorar
    """
    print("\n" + "="*80)
    print("🎯 EXPLORACIÓN DE VARIABLES CLAVE")
    print("="*80)
    
    # Variable objetivo principal: Company response
    if "Company response" in df.columns:
        print("\n🎯 VARIABLE OBJETIVO: Company response")
        response_counts = df["Company response"].value_counts()
        response_pct = df["Company response"].value_counts(normalize=True) * 100
        
        response_summary = pd.DataFrame({
            "Respuesta": response_counts.index,
            "Cantidad": response_counts.values,
            "Porcentaje": response_pct.values.round(2)
        })
        print(response_summary.to_string(index=False))
    
    # Productos más comunes
    if "Product" in df.columns:
        print("\n📦 PRODUCTOS FINANCIEROS:")
        product_counts = df["Product"].value_counts().head(10)
        for product, count in product_counts.items():
            pct = (count / len(df)) * 100
            print(f"   • {product}: {count:,} ({pct:.1f}%)")
    
    # Estados con más quejas
    if "State" in df.columns:
        print("\n🌎 ESTADOS CON MÁS QUEJAS:")
        state_counts = df["State"].value_counts().head(10)
        for state, count in state_counts.items():
            pct = (count / len(df)) * 100
            print(f"   • {state}: {count:,} ({pct:.1f}%)")
    
    # Análisis temporal
    if "Date received" in df.columns:
        print("\n📅 ANÁLISIS TEMPORAL:")
        df_temp = df.copy()
        df_temp["Date received"] = pd.to_datetime(df_temp["Date received"])
        
        date_range = f"{df_temp['Date received'].min().strftime('%Y-%m-%d')} a {df_temp['Date received'].max().strftime('%Y-%m-%d')}"
        print(f"   • Rango de fechas: {date_range}")
        
        # Quejas por año
        df_temp["Year"] = df_temp["Date received"].dt.year
        yearly_counts = df_temp['Year'].value_counts().sort_index()
        print("   • Quejas por año:")
        for year, count in yearly_counts.items():
            print(f"     - {year}: {count:,}")