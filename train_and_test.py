import collections
import pathlib
import random

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
        by_paid[paid] = (all_data[all_data['not.fully.paid'] == paid]).copy()
        print('paid', paid, len(by_paid[paid].index))

    # Mark as train, test, or valid.
    split_ranges = {
        'valid': (-1e-6, 0.05),
        'test': (0.05, 0.2),
        'train': (0.2, 1.)
    }

    concat_frames = []
    for key, value in by_paid.items():
        split_column = []
        for row_id in range(len(value.index)):
            ran_value = random.uniform(0, 1)
            found_split = False
            for split_key, split_value in split_ranges.items():
                if ((ran_value > split_value[0])
                        and (ran_value <= split_value[1])):
                    split_column.append(split_key)
                    found_split = True
                    break
            assert found_split, 'Failed to find split'
        assert len(split_column) == len(value.index)
        value['split'] = split_column
        concat_frames.append(value)

    all_with_splits = pd.concat(concat_frames)
    print('total', len(all_with_splits))

    table_data = collections.defaultdict(list)
    for paid in (0, 1):
        for split in set(all_with_splits['split']):
            sub_data = all_with_splits[(all_with_splits['not.fully.paid'] == paid) &
                                       (all_with_splits['split'] == split)]
            table_data['paid'].append(paid)
            table_data['split'].append(split)
            table_data['count'].append(len(sub_data.index))
    table = pd.DataFrame(table_data)
    print(table)
    


runit()
