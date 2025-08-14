import geopandas as gpd
import ipywidgets as widgets
from ipyleaflet import Map, Choropleth, LayersControl
import json


# Create interactive choropleth
choropleth = Choropleth(
    geo_data=geo_json_data,
    value_property='incident_count',
    colors=['lightblue', 'blue', 'darkblue'],
    colormap=None,
    threshold_scale=[min_count, max_count/4, max_count/2, 3*max_count/4, max_count],
    style={'fillOpacity': 0.7, 'color': 'grey', 'weight': 0.5},
    hover_style={'fillOpacity': 0.9}
)

# Center map on UK
m = Map(center=[55, -3], zoom=5)
m.add_layer(choropleth)
m.add_control(LayersControl())

# Wrap in a widget so it can be saved
widget = widgets.VBox([m])
widget
import ipywidgets as widgets
import ipywidgets.embed as embed

# Save the widget to HTML (Streamlit can load this quickly)
embed.embed_minimal_html('uk_incidents_widget.html', views=[widget])
