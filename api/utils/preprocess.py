# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NADropper(BaseEstimator, TransformerMixin):
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X: pd.DataFrame):
        return X.dropna()


class ColumnsDropper(BaseEstimator, TransformerMixin):
    """Support class to drop specific columns in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.columns)


class DataFrameCaster(BaseEstimator, TransformerMixin):
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)
