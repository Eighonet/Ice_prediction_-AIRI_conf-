import shapely.geometry
import numpy as np
import plotly
import json
import math
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

mapbox_token = "sk.eyJ1IjoiZ29yY2hha292diIsImEiOiJjbDV5NWhiYmEwMTUxM2ptYXR4ZHJxZDE5In0.rkVGQw0UehnOJMlXKZpMCA"

def barents_map(model, date):
    print(model, date)
    print('start barents')

    lats = np.linspace(71,80,500)
    lons = np.linspace(36,41,500)
    magns = np.random.rand(0, 1, 250000)

    fig = go.Figure()
    fig = go.Figure(go.Densitymapbox(lat=lats, lon=lons, z=magns,
                                 radius=10))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=38.62)
    fig.update_layout(margin={"r":30,"t":0,"l":0,"b":0})


    fig.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(
            accesstoken=mapbox_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=75.75,
                lon=38.62
            ),
            pitch=0,
            zoom=3
        ),
        showlegend=True
    )
    print('end barents')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
