"""
Train a model to predict loan payment using the train and valid splits.
Test the model using the test split.
"""
import collections
import pathlib
import sqlite3
from typing import Dict, Optional, Tuple
from numbers import Number

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# mypy: disable-error-code=import
# pylint: disable=consider-using-enumerate


class NetworkHandler:
    """Simple multi-layer dense network."""
    num_dense_layers = 1
    dense_neurons = 3

    def __init__(self, num_features, split_column, gt_column,
                 col_names: Tuple[str, ...]):
        self.num_features = num_features
        self.split_column = split_column
        self.gt_column = gt_column
        self.input_layer = None
        self.output_layer = None
        self.model: Optional[tf.keras.Model] = None
        self.col_index = []
        for col_name in col_names:
            if col_name not in ((split_column, gt_column)):
                self.col_index.append(col_name)
        self.col_index.sort()

    def build_network(self):
        """Build the network."""
        self.input_layer = tf.keras.Input((self.num_features, ))
        layer = self.input_layer
        for i in range(self.num_dense_layers):
            layer = tf.keras.layers.Dense(self.dense_neurons)(layer)
            if i != self.num_dense_layers - 1:
                layer = tf.keras.activations.tanh(layer)
        self.output_layer = tf.keras.activations.softmax(layer)
        self.model = tf.keras.Model(inputs=(self.input_layer, ),
                                    outputs=(self.output_layer, ))

    def get_feature_data(self, split: str,
                         raw_data: pd.DataFrame) -> np.ndarray:
        """Extract the features for each row in the dataframe and
        return them in a numpy array."""
        split_data = raw_data[raw_data['split'] == split]
        result = []
        for row_id in range(len(split_data.index)):
            input_row = split_data.iloc[row_id]
            result_row = []
            for i in range(len(self.col_index)):
                result_row.append(input_row[self.col_index[i]])
            result.append(result_row)
        return np.array(result)

    def get_gt_data(self, split: str, raw_data: pd.DataFrame) -> np.ndarray:
        """Extract the ground truth for each row in the dataframe and
        return them in a numpy array."""
        split_data = raw_data[raw_data['split'] == split]
        result = []
        for row_id in range(len(split_data.index)):
            input_row = split_data.iloc[row_id]
            result.append(input_row[self.gt_column])
        return np.array(result)

    def train_network(self,
                      train_valid_data: pd.DataFrame,
                      num_epochs: int = 50):
        """Train the network using train and valid splits."""

        sample_counts: Dict[Number, int] = collections.defaultdict(int)
        for row_id in range(len(train_valid_data.index)):
            row = train_valid_data.iloc[row_id]
            sample_counts[row[self.gt_column]] += 1
        sample_fractions = {}
        for key in sample_counts:
            sample_fractions[key] = float(sample_counts[key]) / len(
                train_valid_data.index)
        # Class weights are the inverse of the sample counts.
        class_weights: Dict[Number, float] = {}
        for key, value in sample_fractions.items():
            class_weights[key] = 1. / value
        x_data = self.get_feature_data('train', train_valid_data)
        y_data = self.get_gt_data('train', train_valid_data)
        validation_data = (self.get_feature_data('valid', train_valid_data),
                           self.get_gt_data('valid', train_valid_data))
        assert self.model is not None
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=[tf.keras.metrics.BinaryAccuracy()])
        self.model.fit(x_data,
                       y_data,
                       validation_data=validation_data,
                       epochs=num_epochs,
                       class_weight=class_weights)

    def test_network(self, test_data: pd.DataFrame):
        """Test the network using the test split."""
        x_data = self.get_feature_data('test', test_data)
        y_data = self.get_gt_data('test', test_data)
        if self.model is not None:
            result = self.model.evaluate(x_data, y_data)
            print('evaluate')
            print(result)


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
        self.transformer = QuantileTransformer(n_quantiles=200)
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
        transformed_data = transformed_data.reshape(
            (transformed_data.shape[0], ))
        data[self.feature_name] = transformed_data


def runit():
    """main program"""
    # Load the data from the database.
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from loan_data', conn)
    special_columns = ('split', 'not_fully_paid')

    # Transform the features with quantile mapping, but skip
    # special columns.
    for col_name in all_data.columns:
        if col_name in special_columns:
            continue
        normalizer = FeatureNormalizer(col_name, special_columns)
        normalizer.train(all_data[all_data['split'] == 'train'])
        print(col_name, normalizer.get_status())
        normalizer.normalize(all_data)

    network = NetworkHandler(
        len(all_data.columns) - len(special_columns), 'split',
        'not_fully_paid', all_data.columns)
    network.build_network()
    network.train_network(all_data, 10)
    network.test_network(all_data)


runit()
