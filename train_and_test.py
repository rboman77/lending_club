import pathlib

import numpy as np
import tensorflow as tf
import pandas as pd

# mypy: disable-error-code=import


def runit():
    data_folder = pathlib.Path(__file__).resolve()
    print(data_folder)


runit()
