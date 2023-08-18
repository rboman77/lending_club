"""
Plot results from loan prediction.
"""

# mypy: disable-error-code=import
# pylint: disable=unused-import

import pathlib
import sqlite3

import pandas as pd
import hvplot.pandas  # noqa
import holoviews as hv


def runit():
    """main program"""
    # Load the prediction data from the database.
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from predicted_loans', conn)
    test_data = all_data[all_data['split'] == 'test']
    hv.extension('bokeh')
    plot = test_data.hvplot.hist('prediction', by='not_fully_paid')
    hv.save(plot, data_folder / 'histogram_plot.png')


runit()
