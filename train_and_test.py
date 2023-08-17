"""
Train a model to predict loan payment using the train and valid splits.
Test the model using the test split.
"""
import pathlib
import sqlite3
from typing import Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# mypy: disable-error-code=import


class NetworkHandler:
    """Simple multi-layer dense network."""
    num_dense_layers = 4
    dense_neurons = 4

    def __init__(self, num_features, split_column, gt_column):
        self.num_features = num_features
        self.split_column = split_column
        self.gt_column = gt_column
        self.input_layer = None
        self.output_layer = None
        self.model = None

    def build_network(self):
        """Build the network."""
        self.input_layer = tf.keras.Input((self.num_features, ))
        layer = self.input_layer
        for _ in range(self.num_dense_layers):
            layer = tf.keras.layers.Dense(self.dense_neurons)(layer)
        self.output_layer = layer
        self.model = tf.keras.Model(inputs=(self.input_layer, ),
                                    outputs=(self.output_layer, ))

    def train_network(self, train_data: pd.DataFrame):
        """Train the network using train and valid splits."""

    def test_network(self):
        """Test the network using the test split."""


class FeatureNormalizer:
    """Normalize a single feature in a dataset.

       binary feature: leave as-is.
       categorical feature: replace with index.
       numeric feature: normalize with quantile transform
    """

    def __init__(self, feature_name: str, special_columns: Tuple[str, ...]):
        self.feature_name = feature_name
        if self.feature_name in special_columns:
            self.status = 'no transform'
        else:
            self.status = 'not trained'
        self.transformer = QuantileTransformer(n_quantiles=50)
        self.category_index: Tuple[str, ...] = ('', )

    def get_status(self):
        """Return status string."""
        return self.status

    def extract_feature(self, table: pd.DataFrame) -> np.ndarray:
        """Extract column of features from table."""
        result = np.array(table[self.feature_name])
        num_rows = result.shape[0]
        result = result.reshape((num_rows, 1))
        return result

    def train(self, train_data: pd.DataFrame):
        """Train the mapping."""
        value_set = set(train_data[self.feature_name])
        value_list = list(value_set)
        first_value = list(value_set)[0]
        if isinstance(first_value, str):
            self.status = 'categorical'
            value_list.sort()
            self.category_index = tuple(value_list)
            return

        if value_set == {0, 1}:
            self.status = 'binary'
            return

        # Use quantile transform.
        self.transformer.fit(self.extract_feature(train_data))
        self.status = 'quantile'

    def normalize(self, data: pd.DataFrame):
        """Apply the trained mapping to the data."""
        assert self.status != 'not trained', 'Transformer not trained'
        if self.status == 'binary':
            return

        if self.status == 'categorical':
            trans_data = []
            for row_id in range(len(data.index)):
                value = data.iloc[row_id][self.feature_name]
                trans_data.append(
                    self.category_index.index(value) /
                    len(self.category_index))
            data[self.feature_name] = trans_data
            return

        # Sanity check.
        assert self.status == 'quantile'

        transformed_data = self.transformer.transform(
            self.extract_feature(data))
        trans_data = transformed_data.reshape((transformed_data.shape[0], ))


def runit():
    """main program"""
    # Load the data from the database.
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from loan_data', conn)
    special_columns = ('split', 'not_fully_paid')

    print('befor normalize')
    print(all_data.iloc[:10])
    # Transform the features with quantile mapping, but skip
    # special columns and binary features.
    for col_name in all_data.columns:
        if col_name in special_columns:
            continue
        print('before', col_name)
        print(all_data[col_name])
        normalizer = FeatureNormalizer(col_name, special_columns)
        normalizer.train(all_data[all_data['split'] == 'train'])
        print(col_name, normalizer.get_status())
        normalizer.normalize(all_data)
        print('after')
        print(all_data[col_name])
    print('after normalize')
    print(all_data.iloc[:10])

    network = NetworkHandler(
        len(all_data.columns) - len(special_columns), 'split',
        'not_fully_paid')
    network.build_network()
    print('network built')


runit()
