from ipyleaflet import Map, CircleMarker, LayerGroup, WidgetControl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import ipywidgets as widgets
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import base64
import ipywidgets as widgets
from ipyleaflet import WidgetControl

def crear_leyenda_magma(min_val, max_val):
    # Crear figura horizontal
    fig, ax = plt.subplots(figsize=(4, 0.4))
    cmap = plt.get_cmap("magma")
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

    ax.set_title('Cluster ID', fontsize=10)
    cb.set_ticks(range(min_val, max_val + 1))
    cb.ax.tick_params(labelsize=8)

    # Guardar como imagen base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_b64


def mostrar_ositas_cluster(cluster_summary):
    m = Map(
        # center=(83, -40),
        zoom=1.3,
        scroll_wheel_zoom=True,
        dragging=True,
        zoom_control=False,
        world_copy_jump=True
    )

    # Asegurar columnas en minúsculas
    cluster_summary.columns = cluster_summary.columns.str.lower()
    cluster_summary = cluster_summary.reset_index()  # asegurar que 'cluster' sea columna

    # Elegimos un colormap más alegre 
    cmap = cm.get_cmap("magma", len(cluster_summary))
    norm = mcolors.Normalize(vmin=0, vmax=len(cluster_summary) - 1)

    # Escalado del radio en función de uniqueanimalid
    min_count = cluster_summary["uniqueanimalid"].min()
    max_count = cluster_summary["uniqueanimalid"].max()

    # Crear marcadores con colores y tamaños por cluster
    marcadores = []
    leyenda_items = []

    for i, row in cluster_summary.iterrows():
        cluster_id = row["cluster"]
        lat = row["avglatitude"]
        lon = row["avglongitude"]
        count = row["uniqueanimalid"]

        color = mcolors.to_hex(cmap(norm(i)))
        radius = np.interp(count, (min_count, max_count), (1, 8))

        marcador = CircleMarker(
            location=(lat, lon),
            radius=int(radius),
            color=color,
            fill_color=color,
            fill_opacity=0.75
        )
        marcadores.append(marcador)

        leyenda_items.append(
            f"<div style='color:{color}; font-size:14px; line-height:1.4;'><b>■</b>🐾 Cluster {cluster_id} ({count} osas)</div>"
        )

    # Añadir capa al mapa
    m.add_layer(LayerGroup(layers=marcadores))

    # Crear leyenda con HTML estilizado
    # leyenda_html = widgets.HTML(
    #     "<div style='padding: 8px; background-color: white; border:1px solid #ccc;'>"
    #     "<b>🐾 Clusters de osas polares:</b><br>" + "<br>".join(leyenda_items) + "</div>"
    # )
    # leyenda = WidgetControl(widget=leyenda_html, position="bottomright")
    # m.add_control(leyenda)

    # Rango de cluster IDs
    min_cluster = int(cluster_summary["cluster"].min())
    max_cluster = int(cluster_summary["cluster"].max())

    # Crear imagen de leyenda
    img_b64 = crear_leyenda_magma(min_cluster, max_cluster)
    leyenda_img = widgets.HTML(
        value=f"<img src='data:image/png;base64,{img_b64}' style='width:250px;'>"
    )
    leyenda_widget = WidgetControl(widget=leyenda_img, position='bottomright')
    m.add_control(leyenda_widget)


    return m
