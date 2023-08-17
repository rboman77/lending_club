"""
Train a model to predict loan payment using the train and valid splits.
Test the model using the test split.
"""
import pathlib
import sqlite3

import tensorflow as tf
import pandas as pd

# mypy: disable-error-code=import


def runit():
    """main program"""
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from loan_data', conn)
    print(all_data)


runit()
