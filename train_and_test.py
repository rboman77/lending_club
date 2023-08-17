"""
Train a model to predict loan payment using the train and valid splits.
Test the model using the test split.
"""
import pathlib
import sqlite3

import tensorflow as tf
import pandas as pd

# mypy: disable-error-code=import


class NetworkHandler:
    """Simple multi-layer dense network."""
    num_dense_layers = 4
    dense_neurons = 4

    def __init__(self, num_features):
        self.num_features = num_features
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


def runit():
    """main program"""
    # Load the data from the database.
    code_folder = pathlib.Path(__file__).resolve().parents[0]
    data_folder = code_folder.parents[0] / 'data'
    conn = sqlite3.connect(data_folder / 'lending_club_loan.sqlite')
    all_data = pd.read_sql_query('select * from loan_data', conn)
    # 1 column is ground truth, another column is split.
    # The rest are features.
    network = NetworkHandler(len(all_data.columns) - 1)
    network.build_network()


runit()
