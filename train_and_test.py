import pathlib

import numpy as np
import tensorflow as tf
import pandas as pd

# mypy: disable-error-code=import


def runit():
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    all_data = pd.read_csv(data_folder / 'lending_club_loan_data_analysis.csv')
    # Split data by paid and not paid.
    by_paid = {}
    for paid in (1, 0):
        by_paid[paid] = all_data[all_data['not.fully.paid'] == paid]
        print('paid', paid, len(by_paid[paid].index))


runit()
