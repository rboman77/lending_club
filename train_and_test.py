import pathlib

import numpy as np
import tensorflow as tf
import pandas as pd

# mypy: disable-error-code=import


def runit():
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    print('code folder', code_folder)
    print('data folder', data_folder)
    all_data = pd.read_csv(data_folder / 'lending_club_loan_data_analysis.csv')
    print(all_data)


runit()
