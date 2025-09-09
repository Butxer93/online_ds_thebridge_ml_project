import matplotlib.pyplot as plt

def create_initial_visualizations(df):
    """
    Crea visualizaciones iniciales para entender los datos

    Parameters:
    -----------
    df : pd.DataFrame
    DataFrame a visualizar
    """
    print("\n" + "="*80)
    print("📊 CREANDO VISUALIZACIONES INICIALES")
    print("="*80)
    
    # Configurar subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Exploración Inicial de Datos - Quejas Financieras", fontsize=16, fontweight="bold")

    # 1. Distribución de respuestas de empresa
    if "Company response" in df.columns:
        ax1 = axes[0, 0]
        response_counts = df["Company response"].value_counts()
        ax1.pie(response_counts.values, labels=response_counts.index, autopct='%1.1f%%')
        ax1.set_title("Distribución de Respuestas de Empresa", fontweight="bold")

    # 2. Top 10 productos
    if "Product" in df.columns:
        ax2 = axes[0, 1]
        top_products = df["Product"].value_counts().head(10)
        ax2.barh(range(len(top_products)), top_products.values)
        ax2.set_yticks(range(len(top_products)))
        ax2.set_yticklabels(top_products.index, fontsize=8)
        ax2.set_title("Top 10 Productos Más Reportados", fontweight="bold")
        ax2.set_xlabel("Número de Quejas")

    # 3. Estados con más quejas
    if "State" in df.columns:
        ax3 = axes[1, 0]
        top_states = df["State"].value_counts().head(10)
        ax3.bar(range(len(top_states)), top_states.values)
        ax3.set_xticks(range(len(top_states)))
        ax3.set_xticklabels(top_states.index, rotation=45)
        ax3.set_title("Top 10 Estados con Más Quejas", fontweight="bold")
        ax3.set_ylabel("Número de Quejas")

    # 4. Respuesta oportuna
    if "Timely response?" in df.columns:
        ax4 = axes[1, 1]
        timely_counts = df["Timely response?"].value_counts()
        colors = ["lightgreen" if x == "Yes" else "lightcoral" for x in timely_counts.index]
        ax4.bar(timely_counts.index, timely_counts.values, color=colors)
        ax4.set_title("Respuestas Oportunas vs No Oportunas", fontweight="bold")
        ax4.set_ylabel("Número de Casos")

    # Añadir porcentajes
    total = timely_counts.sum()
    for i, (idx, val) in enumerate(timely_counts.items()):
        pct = (val / total) * 100
    ax4.text(i, val + total*0.01, f'{pct:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()