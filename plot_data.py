"""
Plot results from loan prediction.
"""

# mypy: disable-error-code=import

import collections
import pathlib
import sqlite3

import pandas as pd
import holoviews as hv
from sklearn.metrics import roc_curve


def make_roc_curve(table: pd.DataFrame, split: str, predict_label: str,
                   factor: float):
    """Compute ROC curve using sklearn."""
    sub_table = table[table['split'] == split]
    truth_values = []
    predict_values = []
    for row_id in range(len(sub_table.index)):
        row = sub_table.iloc[row_id]
        if row['not_fully_paid'] == 1:
            truth_values.append(1)
        else:
            truth_values.append(0)
        predict_values.append(row[predict_label] * factor)
    true_pos_rate, false_pos_rate, _ = roc_curve(truth_values, predict_values)
    result_data = collections.defaultdict(list)
    for true_p, false_p in zip(true_pos_rate, false_pos_rate):
        result_data['true_positive'].append(true_p)
        result_data['false_positive'].append(false_p)
    return pd.DataFrame(result_data)


def runit():
    """main program"""
    # Load the prediction data from the database.
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    image_folder = code_folder / 'images'
    if not image_folder.exists():
        image_folder.mkdir()
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from predicted_loans', conn)
    hv.extension('matplotlib')
    plot_list = []
    plot_options = {
        'fig_size': 200,
        'show_grid': True,
    }
    # Predict using neural network score.
    predict_scores = make_roc_curve(all_data, 'test', 'prediction', 1.)
    plot = hv.Curve(predict_scores,
                    kdims='true_positive',
                    vdims='false_positive',
                    label='neural network')
    plot_list.append(plot.opts(**plot_options))
    # Predict using FICO score only.
    predict_scores = make_roc_curve(all_data, 'test', 'fico', -1.)
    plot = hv.Curve(predict_scores,
                    kdims='true_positive',
                    vdims='false_positive',
                    label='FICO only')
    plot_list.append(plot.opts(**plot_options))
    plot = hv.Overlay(plot_list)
    plot = plot.opts(legend_position='top_left', title='Bad Loan Prediction')
    hv.save(plot, image_folder / 'roc_curve.svg')


runit()
