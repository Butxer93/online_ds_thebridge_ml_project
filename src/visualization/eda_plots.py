import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Análisis de la variable objetivo
def target_analysis(df):
    """Análisis profundo de la variable objetivo"""
    print("\n📊 ANÁLISIS DE VARIABLE OBJETIVO")
    print("-" * 50)
    
    if "Company response" in df.columns:
        # Distribución de respuestas
        response_dist = df["Company response"].value_counts()
        print("Distribución de respuestas de empresa:")
        for response, count in response_dist.items():
            pct = (count / len(df)) * 100
            print(f"   • {response}: {count:,} ({pct:.1f}%)")
        
        # Crear visualización
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Análisis de Variable Objetivo: Company Response", fontsize=16, fontweight="bold")
        
        # 1. Distribución general
        response_dist.plot(kind="pie", ax=axes[0,0], autopct="%1.1f%%")
        axes[0,0].set_title("Distribución de Respuestas")
        axes[0,0].set_ylabel("")
        
        # 2. Por región (si existe)
        if "region" in df.columns:
            pd.crosstab(df["region"], df["Company response"]).plot(kind="bar", ax=axes[0,1])
            axes[0,1].set_title("Respuestas por Región")
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[0,1].tick_params(axis="x", rotation=45)
        
        # 3. Por año
        if "year_received" in df.columns:
            yearly_response = pd.crosstab(df["year_received"], df["Company response"])
            yearly_response.plot(kind="line", ax=axes[1,0], marker="o")
            axes[1,0].set_title("Tendencia de Respuestas por Año")
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # 4. Tiempo de respuesta vs tipo
        if "Timely response?" in df.columns:
            timely_response = pd.crosstab(df["Company response"], df["Timely response?"])
            timely_response.plot(kind="bar", ax=axes[1,1])
            axes[1,1].set_title("Puntualidad por Tipo de Respuesta")
            axes[1,1].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.show()

# 2. Análisis temporal
def temporal_analysis(df):
    """Análisis temporal de las quejas"""
    print("\n📅 ANÁLISIS TEMPORAL")
    print("-" * 50)
    
    if "Date received" in df.columns:
        # Tendencias por año
        yearly_complaints = df.groupby("year_received").size()
        print("Quejas por año:")
        for year, count in yearly_complaints.items():
            print(f"   • {year}: {count:,}")
        
        # Visualizaciones temporales
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Análisis Temporal de Quejas", fontsize=16, fontweight="bold")
        
        # 1. Tendencia anual
        yearly_complaints.plot(kind="line", ax=axes[0,0], marker="o", linewidth=2)
        axes[0,0].set_title("Tendencia Anual de Quejas")
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribución mensual
        monthly_complaints = df.groupby("month_received").size()
        monthly_complaints.plot(kind="bar", ax=axes[0,1], color="skyblue")
        axes[0,1].set_title("Distribución Mensual")
        axes[0,1].set_xlabel("Mes")
        
        # 3. Distribución por día de la semana
        if "dayofweek_received" in df.columns:
            days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
            dayofweek_complaints = df.groupby('dayofweek_received').size()
            dayofweek_complaints.index = days
            dayofweek_complaints.plot(kind="bar", ax=axes[1,0], color="lightcoral")
            axes[1,0].set_title("Distribución por Día de la Semana")
            axes[1,0].tick_params(axis="x", rotation=45)
        
        # 4. Tiempo de procesamiento
        if "processing_days" in df.columns:
            processing_stats = df["processing_days"].describe()
            axes[1,1].hist(df["processing_days"], bins=30, color="lightgreen", alpha=0.7)
            axes[1,1].set_title("Distribución de Días de Procesamiento")
            axes[1,1].axvline(processing_stats["mean"], color="red", linestyle="--", 
                             label=f'Media: {processing_stats["mean"]:.1f} días')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()

# 3. Análisis geográfico
def geographic_analysis(df):
    """Análisis geográfico"""
    print("\n🗺️ ANÁLISIS GEOGRÁFICO")
    print("-" * 50)
    
    if "State" in df.columns:
        # Top estados con más quejas
        top_states = df['State'].value_counts().head(15)
        print("Top 15 estados con más quejas:")
        for state, count in top_states.items():
            pct = (count / len(df)) * 100
            print(f"   • {state}: {count:,} ({pct:.1f}%)")
        
        # Visualizaciones geográficas
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Análisis Geográfico de Quejas", fontsize=16, fontweight="bold")
        
        # 1. Top 15 estados
        top_states.plot(kind="bar", ax=axes[0,0])
        axes[0,0].set_title("Top 15 Estados por Número de Quejas")
        axes[0,0].tick_params(axis="x", rotation=45)
        
        # 2. Por región
        if "region" in df.columns:
            region_complaints = df["region"].value_counts()
            region_complaints.plot(kind="pie", ax=axes[0,1], autopct='%1.1f%%')
            axes[0,1].set_title("Distribución por Región")
            axes[0,1].set_ylabel("")
        
        # 3. Heatmap: Estado vs Tipo de Respuesta
        if "Company response" in df.columns:
            state_response = pd.crosstab(df["State"], df["Company response"])
            top_10_states = df["State"].value_counts().head(10).index
            state_response_top = state_response.loc[top_10_states]
            
            sns.heatmap(state_response_top, annot=True, fmt="d", ax=axes[1,0], cmap="YlOrRd")
            axes[1,0].set_title("Heatmap: Top 10 Estados vs Tipo de Respuesta")
            axes[1,0].tick_params(axis="x", rotation=45)
        
        # 4. Quejas per capita (simulado)
        if "state_complaint_count" in df.columns:
            state_stats = df.groupby("State")["state_complaint_count"].first().sort_values(ascending=False).head(10)
            state_stats.plot(kind="bar", ax=axes[1,1], color="orange")
            axes[1,1].set_title("Top 10 Estados: Total de Quejas")
            axes[1,1].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plt.show()

# 4. Análisis de productos y empresas
def product_company_analysis(df):
    """Análisis de productos y empresas"""
    print("\n🏢 ANÁLISIS DE PRODUCTOS Y EMPRESAS")
    print("-" * 50)
    
    # Análisis de productos
    if "Product" in df.columns:
        top_products = df["Product"].value_counts().head(10)
        print("Top 10 productos con más quejas:")
        for product, count in top_products.items():
            pct = (count / len(df)) * 100
            print(f"   • {product}: {count:,} ({pct:.1f}%)")
    
    # Análisis de empresas
    if "Company" in df.columns:
        top_companies = df["Company"].value_counts().head(10)
        print(f"\nTop 10 empresas con más quejas:")
        for company, count in top_companies.items():
            pct = (count / len(df)) * 100
            print(f"   • {company}: {count:,} ({pct:.1f}%)")
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Análisis de Productos y Empresas", fontsize=16, fontweight="bold")
    
    # 1. Top productos
    if "Product" in df.columns:
        top_products.plot(kind="barh", ax=axes[0,0])
        axes[0,0].set_title("Top 10 Productos Más Reportados")
        axes[0,0].set_xlabel("Número de Quejas")
    
    # 2. Categorías de productos
    if "product_category" in df.columns:
        category_dist = df["product_category"].value_counts()
        category_dist.plot(kind="pie", ax=axes[0,1], autopct="%1.1f%%")
        axes[0,1].set_title("Distribución por Categoría de Producto")
        axes[0,1].set_ylabel("")
    
    # 3. Top empresas
    if "Company" in df.columns:
        top_companies.plot(kind="barh", ax=axes[1,0])
        axes[1,0].set_title("Top 10 Empresas Más Reportadas")
        axes[1,0].set_xlabel("Número de Quejas")
    
    # 4. Tamaño de empresa vs respuesta
    if all(col in df.columns for col in ["company_size", "Company response"]):
        size_response = pd.crosstab(df["company_size"], df["Company response"])
        size_response.plot(kind="bar", ax=axes[1,1])
        axes[1,1].set_title("Tamaño de Empresa vs Tipo de Respuesta")
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1,1].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Análisis cruzado: Producto vs Respuesta
    if all(col in df.columns for col in ['Product', 'Company response']):
        print(f"\n📊 ANÁLISIS CRUZADO: PRODUCTO VS RESPUESTA")
        product_response = pd.crosstab(df['Product'], df['Company response'], normalize='index') * 100
        
        # Mostrar solo los productos más comunes
        top_5_products = df["Product"].value_counts().head(5).index
        product_response_top = product_response.loc[top_5_products]
        
        print("Porcentaje de tipos de respuesta por producto (Top 5):")
        print(product_response_top.round(2))
        
        # Visualización del análisis cruzado
        plt.figure(figsize=(12, 8))
        sns.heatmap(product_response_top, annot=True, fmt=".1f", cmap="RdYlBu_r")
        plt.title("Heatmap: Producto vs Tipo de Respuesta (%)", fontsize=14, fontweight="bold")
        plt.ylabel("Producto")
        plt.xlabel("Tipo de Respuesta")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 5. Análisis de correlaciones
def correlation_analysis(df):
    """Análisis de correlaciones"""
    print("\n🔗 ANÁLISIS DE CORRELACIONES")
    print("-" * 50)
    
    # Seleccionar variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 2:
        print(f"Variables numéricas encontradas: {len(numeric_cols)}")
        for col in numeric_cols[:10]:  # Mostrar solo las primeras 10
            print(f"   • {col}")
        
        # Calcular correlaciones
        corr_matrix = df[numeric_cols].corr()
        
        # Visualización de correlaciones
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                   cmap="coolwarm", center=0, square=True)
        plt.title("Matriz de Correlaciones - Variables Numéricas", 
                 fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
        
        # Correlaciones más fuertes
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Solo correlaciones > 0.3
                    corr_pairs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        if corr_pairs:
            print(f"\nCorrelaciones significativas (|r| > 0.3):")
            corr_df = pd.DataFrame(corr_pairs).sort_values("correlation", key=abs, ascending=False)
            for _, row in corr_df.head(10).iterrows():
                print(f"   • {row['var1']} ↔ {row['var2']}: {row['correlation']:.3f}")
    
    # Análisis de asociación para variables categóricas
    categorical_cols = ["Product", "Company response", "Timely response?", "region"]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if len(categorical_cols) >= 2:
        print(f"\n📊 ANÁLISIS DE ASOCIACIÓN - VARIABLES CATEGÓRICAS")
        
        # Chi-cuadrado entre variables categóricas importantes
        if all(col in df.columns for col in ["Product", "Company response"]):
            from scipy.stats import chi2_contingency
            
            contingency = pd.crosstab(df["Product"], df["Company response"])
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            print(f"Test Chi-cuadrado: Producto vs Respuesta")
            print(f"   • Chi2: {chi2:.3f}")
            print(f"   • p-value: {p_value:.6f}")
            print(f"   • Asociación: {'Significativa' if p_value < 0.05 else 'No significativa'}")

def advanced_eda(df):
    """
    Análisis exploratorio de datos avanzado
    """
    print("\n" + "="*80)
    print("🔍 ANÁLISIS EXPLORATORIO AVANZADO")
    print("="*80)
    
    target_analysis(df)
    
    temporal_analysis(df)
    
    geographic_analysis(df)
    
    product_company_analysis(df)
    
    correlation_analysis(df)

def detect_outliers(df):
    """Detección de valores atípicos"""
    print("\n" + "="*80)
    print("🔍 DETECCIÓN DE OUTLIERS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("No se encontraron columnas numéricas para análisis de outliers")
        return df
    
    outlier_info = []
    
    for col in numeric_cols:
        if col in ["Complaint ID"]:  # Saltar IDs
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_info.append({
            "column": col,
            "outliers": outlier_count,
            "percentage": outlier_pct,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })
    
    # Mostrar resumen de outliers
    outlier_df = pd.DataFrame(outlier_info)
    outlier_df = outlier_df[outlier_df["outliers"] > 0].sort_values("percentage", ascending=False)
    
    if len(outlier_df) > 0:
        print("Outliers detectados por columna:")
        print(outlier_df.to_string(index=False, float_format="%.2f"))
        
        # Visualizar outliers
        cols_to_plot = outlier_df.head(4)["column"].tolist()
        if cols_to_plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            fig.suptitle("Detección de Outliers - Boxplots", fontsize=16, fontweight="bold")
            
            for i, col in enumerate(cols_to_plot):
                if i < 4:
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'{col}\n({outlier_df[outlier_df["column"]==col]["outliers"].iloc[0]} outliers)')
            
            # Ocultar subplots vacíos
            for i in range(len(cols_to_plot), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
    else:
        print("✅ No se detectaron outliers significativos")
    
    return outlier_df