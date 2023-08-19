"""
Train a model to predict loan payment using the train and valid splits.
Run inference and save the results in the database.
"""
import collections
import pathlib
import sqlite3
from typing import Dict, Optional, Tuple, List
from numbers import Number

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# mypy: disable-error-code=import
# pylint: disable=consider-using-enumerate


class NetworkHandler:
    """Simple multi-layer dense network."""
    num_dense_layers = 4
    dense_neurons = 20

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
        for _ in range(self.num_dense_layers):
            layer = tf.keras.layers.Dense(self.dense_neurons)(layer)
            layer = tf.keras.activations.relu(layer)
        layer = tf.keras.layers.Dense(1)(layer)
        self.output_layer = layer
        self.model = tf.keras.Model(inputs=(self.input_layer, ),
                                    outputs=(self.output_layer, ))

    def get_feature_data(self, split: str,
                         raw_data: pd.DataFrame) -> np.ndarray:
        """Extract the features for each row in the dataframe and
        return them in a numpy array."""
        if split == 'all':
            split_data = raw_data
        else:
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
        class_weights: Dict[Number, float] = {}
        max_weight = 0.
        # Class weights are inversely proportional to count.
        for key, value in sample_counts.items():
            class_weights[key] = 1. / value
            max_weight = max(max_weight, class_weights[key])
        for key in class_weights.keys():
            class_weights[key] /= max_weight
        x_data = self.get_feature_data('train', train_valid_data)
        y_data = self.get_gt_data('train', train_valid_data)
        validation_data = (self.get_feature_data('valid', train_valid_data),
                           self.get_gt_data('valid', train_valid_data))
        assert self.model is not None
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=[tf.keras.metrics.AUC()])
        self.model.fit(x_data,
                       y_data,
                       validation_data=validation_data,
                       epochs=num_epochs,
                       class_weight=class_weights,
                       callbacks=tf.keras.callbacks.EarlyStopping(patience=5))

    def network_description(self):
        """Print the network description."""
        self.model.summary()

    def evaluate_network(self, test_data: pd.DataFrame):
        """Test the network using the test split."""
        x_data = self.get_feature_data('test', test_data)
        y_data = self.get_gt_data('test', test_data)
        if self.model is not None:
            result = self.model.evaluate(x_data, y_data)
            print('evaluate')
            print(result)

    def predict(self, test_data: pd.DataFrame):
        """Run inference and save the predictions in the data frame."""
        x_data = self.get_feature_data('all', test_data)
        if self.model is not None:
            predictions = self.model.predict(x_data)
            predict_shape = predictions[0].shape
            print('predictions shape', predict_shape)
            test_data['prediction'] = list(predictions[0].reshape(
                (predict_shape[0], )))


class FeatureNormalizer:
    """Normalize a single feature in a dataset.

       binary feature: leave as-is.
       categorical feature: add one-hot columns
       numeric feature: normalize with quantile transform
    """

    def __init__(self, feature_name: str, special_columns: Tuple[str, ...]):
        self.feature_name = feature_name
        if self.feature_name in special_columns:
            self.status = 'no transform'
        else:
            self.status = 'not trained'
        self.transformer = QuantileTransformer(n_quantiles=20)
        self.category_list: List[str] = []

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
            self.category_list = value_list
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
            # For each row in data, make a one-hot representation in
            # new column dictionaries.
            transformed_columns: Dict[str,
                                      list] = collections.defaultdict(list)
            for row_id in range(len(data.index)):
                value = data.iloc[row_id][self.feature_name]
                for i, category_value in enumerate(self.category_list):
                    # Make one-hot representation of
                    col_name = f'{self.feature_name}_{i}'
                    if category_value == value:
                        transformed_columns[col_name].append(1.)
                    else:
                        transformed_columns[col_name].append(0.)
            # Remove the original column.
            data.drop(self.feature_name, axis=1, inplace=True)
            # Insert the added columns.
            for key, value in transformed_columns.items():
                data.insert(0, key, value)
            return

        # If we get here, the status should be quantile..
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
    print('train samples', len(all_data[all_data['split'] == 'train']))
    special_columns = ('split', 'not_fully_paid')

    print('before normalize')
    print(all_data.iloc[:10])

    # Normalize the features.
    for col_name in all_data.columns:
        if col_name in special_columns:
            continue
        normalizer = FeatureNormalizer(col_name, special_columns)
        normalizer.train(all_data[all_data['split'] == 'train'])
        normalizer.normalize(all_data)

    print('after normalize')
    print(all_data.iloc[:10])

    network = NetworkHandler(
        len(all_data.columns) - len(special_columns), 'split',
        'not_fully_paid', all_data.columns)
    network.build_network()
    network.network_description()
    network.train_network(all_data, 30)
    network.evaluate_network(all_data)
    network.predict(all_data)
    all_data.to_sql('predicted_loans', conn, if_exists='replace')


runit()
