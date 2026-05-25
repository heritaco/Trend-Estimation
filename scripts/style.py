def mpl_apply():
    """
    Aplica un estilo personalizado a las gráficas de Matplotlib y Seaborn.
    from style import apply
    mpl_apply()
    """
    import seaborn as sns

    sns.set(
        style="whitegrid",
        palette="muted",
        font_scale=1.5,
        rc={
            "grid.linestyle": "--",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.color": "lightgray",
            "figure.figsize": (16/2, 9/2),
            "axes.titlesize": 16,
            "axes.labelsize": 10,

            # Tick labels: tamaño + color
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",

            # Ticks: asegúrate de que se dibujan
            "xtick.bottom": True,        # mostrar ticks abajo
            "xtick.top": False,
            "ytick.left": True,
            "ytick.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,       # longitud de la marca
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "axes.titlepad": 15,

            # Legend: fondo sí, sin borde visible, esquinas cuadradas
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.8,

            # Spines: solo abajo y a la izquierda, en negro
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,

            # Límites y márgenes
            "axes.autolimit_mode": "round_numbers",
            "axes.xmargin": 0.0,    # sin margen extra
            "axes.ymargin": 0.0,
        },
    )

def set_style():
    mpl_apply()

