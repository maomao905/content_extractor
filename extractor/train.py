import logging
import re
import csv
import numpy as np
import dill
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
import xgboost as xgb

from extractor.util import load_log_config

logging.config.dictConfig(load_log_config())
logger = logging.getLogger('applog.' + __name__)

dill.settings['recurse'] = True

class Extractor(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, data_type):
        self.col_name = col_name
        self.data_type = data_type

    def transform(self, X):
        return np.asarray(X[self.col_name]).astype(self.data_type)

    def fit(self, *_):
        return self

class Apply(BaseEstimator, TransformerMixin):
    """Apply a function f element-wise to the numpy array
    """
    def __init__(self, fn):
        self.fn = np.vectorize(fn)

    def transform(self, data):
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self

def preprocess(attr):
    attr = attr.replace('_', '-')
    attr = re.sub('\d+', '0', attr)
    return attr.lower()

# https://github.com/michelleful/SingaporeRoadnameOrigins/blob/24c5162cc8c544d8dfe220c7001382baeb4b3084/notebooks/04%20Adding%20features%20with%20Pipelines.ipynb
def train(df):
    features = df.drop(['attr_name', 'parent_attr_name', 'score'], axis=1)
    scores = df['score']
    X_train, X_test, y_train, y_true = train_test_split(features, scores, test_size=0.2)
    ngram_counter = CountVectorizer(ngram_range=(3, 5), analyzer='char', preprocessor=preprocess)

    model = Pipeline([
        ('features', FeatureUnion([
            ('attr', Pipeline([
                ('select_column', Extractor('concat_attr_name', str)),
                ('ngram', ngram_counter),
            ])),
            ('title_dist', Pipeline([
                ('select_column', Extractor('title_dist', np.float64)),
                ('apply', Apply(lambda x: x)),
                ('scaler', StandardScaler()),
            ])),
            ('text_density', Pipeline([
                ('select_column', Extractor('text_density', np.float64)),
                ('apply', Apply(lambda x: x)),
                ('scaler', StandardScaler()),
            ])),
            ('is_article', Pipeline([
                ('select_column', Extractor('is_article', np.bool)),
                ('apply', Apply(lambda x: x)),
            ])),
        ])),
        ('clf', xgb.XGBRegressor(
            n_estimators=400,
            min_child_weight=3,
            max_depth=3,
            verbosity=3,
        )),
    ])

    y_test = model.predict(X_test)
    logger.info(np.sqrt(mean_squared_error(y_true, y_test)))

    FILE_MODEL = 'data/model.pkl'
    with open(FILE_MODEL, 'wb') as f:
        dill.dump(model, f)
    logger.info('model saved')
