from flask import Flask, render_template, url_for, request, redirect, send_from_directory, flash, jsonify
from flask_wtf import Form, FlaskForm
from wtforms import SelectField
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import os
import uuid
import datetime
from flask_cors import CORS

from charts import barents_map, show_metrics



app = Flask(__name__)
CORS(app)


@app.route('/')
def report():
    return render_template('report.html',
                           area = ['barents', 'laptev', 'labrador'],
                           model = ['UNet', 'ResNet', 'PredRNN', 'ConvLSTM'],
                           scenario = 1,
                           barents_graphJSON= show_map(),
                           metrics_graphJSON= show_hist()
                           )

@app.route('/show_map', methods=['POST', 'GET'])
def show_map():
    return barents_map(request.args.get('model'), request.args.get('date'))

@app.route('/show_hist', methods=['POST', 'GET'])
def show_hist():
    return show_metrics()

# @app.route('/barents_map_')
# def barents_map():
#     return barents_sea_map()
#
# @app.route('/laptev_map_')
# def laptev_map():
#     return laptev_sea_map()
#
# @app.route('/labrador_map_')
# def labrador_map():
#     return labrador_sea_map()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8501)
    # app.run(debug=True)