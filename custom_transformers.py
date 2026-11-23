"""
Custom transformers for the preparation and training notebooks.
Kept in a .py file so preprocessor.pkl can be loaded in other notebooks.
"""

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin

# text helpers from exploration
from squirrel_pipeline import (  # noqa: F401
    ConceptSimilarityTransformer,
    SpacyVectorTransformer,
    concept_similarities,
    discover_anchor_candidates,
    get_nlp,
    probe_anchor,
    texts_to_vectors,
)

set_config(transform_output="pandas")


def _as_frame(X):
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _is_true(col):
    # bools or "true"/"false" strings -> 0/1, missing -> 0
    s = col.astype("string").str.strip().str.lower()
    return s.eq("true").fillna(False).astype(int)


def shift_is_pm(X):
    # AM/PM -> is_pm
    X = _as_frame(X)
    s = X.iloc[:, 0].astype("string").str.strip().str.upper()
    return pd.DataFrame({"is_pm": s.eq("PM").fillna(False).astype(int)}, index=X.index)


def is_pm_name(transformer, input_features):
    return np.array(["is_pm"])


class HectareGridTransformer(BaseEstimator, TransformerMixin):
    """'14E' -> hectare_row (1-42, S to N) and hectare_col (1-9, W to E)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_frame(X)
        parts = X.iloc[:, 0].astype("string").str.strip().str.upper().str.extract(r"^(\d+)([A-I])$")
        row = pd.to_numeric(parts[0], errors="coerce")
        col = parts[1].map({c: i for i, c in enumerate("ABCDEFGHI", start=1)})
        return pd.DataFrame(
            {"hectare_row": row.astype(float), "hectare_col": pd.to_numeric(col, errors="coerce").astype(float)},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(["hectare_row", "hectare_col"])


class PresenceFlag(BaseEstimator, TransformerMixin):
    """1 if the column has a value, else 0 (e.g. has_highlights)"""

    def __init__(self, feature_name="has_value"):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_frame(X)
        s = X.iloc[:, 0].astype("string").str.strip()
        present = s.notna() & s.ne("")
        return pd.DataFrame({self.feature_name: present.fillna(False).astype(int)}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name])


class MissingAsCategory(BaseEstimator, TransformerMixin):
    """Missing and '?' become their own category instead of imputing"""

    def __init__(self, missing_tokens=("?", ""), fill_value="Missing"):
        self.missing_tokens = missing_tokens
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(_as_frame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = _as_frame(X)
        out = {}
        for col in X.columns:
            s = X[col].astype("string").str.strip()
            s = s.mask(s.isin(list(self.missing_tokens)))
            # object dtype so the encoders don't choke on pd.NA
            out[col] = s.fillna(self.fill_value).astype(object)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class BooleanFlags(BaseEstimator, TransformerMixin):
    """Booleans -> 0/1, optional 'any' column (e.g. any_vocalization)"""

    def __init__(self, any_flag_name=None):
        self.any_flag_name = any_flag_name

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(_as_frame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = _as_frame(X)
        out = pd.DataFrame({col: _is_true(X[col]) for col in X.columns}, index=X.index)
        if self.any_flag_name:
            out[self.any_flag_name] = out.max(axis=1)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.any_flag_name:
            names.append(self.any_flag_name)
        return np.array(names, dtype=object)


class InteractionFlags(BaseEstimator, TransformerMixin):
    """Interaction booleans -> 0/1, optionally count 'none checked' as Indifferent"""

    def __init__(self, indifferent_col="Indifferent", merge_none_into_indifferent=True):
        self.indifferent_col = indifferent_col
        self.merge_none_into_indifferent = merge_none_into_indifferent

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(_as_frame(X).columns, dtype=object)
        return self

    def transform(self, X):
        X = _as_frame(X)
        out = pd.DataFrame({col: _is_true(X[col]) for col in X.columns}, index=X.index)
        if self.merge_none_into_indifferent and self.indifferent_col in out.columns:
            none_checked = out.sum(axis=1) == 0
            out.loc[none_checked, self.indifferent_col] = 1
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class SightingHeightTransformer(BaseEstimator, TransformerMixin):
    """Above ground measurement -> feet ('FALSE' = 0, junk = NaN)"""

    def __init__(self, feature_name="above_ground_ft"):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _as_frame(X)
        s = X.iloc[:, 0].astype("string").str.strip().str.upper()
        height = pd.to_numeric(s, errors="coerce")
        height = height.mask(s.eq("FALSE").fillna(False), 0.0)
        return pd.DataFrame({self.feature_name: height.astype(float)}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name])
