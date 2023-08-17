"""
Read the source CSV file.

Rename columns to not have periods to make it easier to work with SQL.

Split into train/test/valid splits so
that there the paid and not paid rows are approximately evenly
split.

Save the data with the split information in sqlite.

Source data comes from
https://www.kaggle.com/datasets/deependraverma13/lending-club-loan-data-analysis-deep-learning
"""

import collections
import pathlib
import random
import sqlite3

import pandas as pd

# mypy: disable-error-code=import


def runit():
    """Main program."""
    # pylint: disable=too-many-locals
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    raw_data = pd.read_csv(data_folder / 'lending_club_loan_data_analysis.csv')

    # Replace periods in column names with underscores.
    all_data = raw_data.rename(columns=lambda x: x.replace('.', '_'))

    # Split data by paid and not paid.
    by_paid = {}
    for paid in (1, 0):
        by_paid[paid] = (all_data[all_data['not_fully_paid'] == paid]).copy()
        print('paid', paid, len(by_paid[paid].index))

    # Use 5% for validation, 15% for test, and 80% for train.
    split_ranges = {
        'valid': (-1e-6, 0.05),
        'test': (0.05, 0.2),
        'train': (0.2, 1.)
    }

    concat_frames = []
    for value in by_paid.values():
        split_column = []
        for _ in range(len(value.index)):
            ran_value = random.uniform(0, 1)
            found_split = False
            for split_key, split_value in split_ranges.items():
                if split_value[0] < ran_value <= split_value[1]:
                    split_column.append(split_key)
                    found_split = True
                    break
            assert found_split, 'Failed to find split'
        assert len(split_column) == len(value.index)
        value['split'] = split_column
        concat_frames.append(value)

    all_with_splits = pd.concat(concat_frames)
    total = float(len(all_with_splits.index))
    print('total', total)

    table_data = collections.defaultdict(list)
    for paid in (0, 1):
        for split in set(all_with_splits['split']):
            sub_data = all_with_splits[
                (all_with_splits['not_fully_paid'] == paid)
                & (all_with_splits['split'] == split)]
            table_data['paid'].append(paid)
            table_data['split'].append(split)
            table_data['count'].append(len(sub_data.index))
            table_data['fraction'].append(len(sub_data.index) / total)
    summary_table = pd.DataFrame(table_data)
    print(summary_table)

    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_with_splits.to_sql('loan_data', conn, if_exists='replace')
    summary_table.to_sql('summary_data', conn, if_exists='replace')


runit()
