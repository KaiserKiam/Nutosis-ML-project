from collections import Counter
import numpy as np
import pandas as pd
import spacy
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

set_config(transform_output="pandas")

_NLP = None


def get_nlp():
    """Load en_core_web_lg once and cache it in memory."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


def texts_to_vectors(texts, exclude_stop=True, exclude_punct=True):
    """Embed each string as spaCy's averaged token vector; empty/NaN -> zero vector."""
    nlp = get_nlp()
    cleaned = pd.Series(texts).fillna("").astype(str).tolist()
    dim = nlp.vocab.vectors_length
    vectors = np.zeros((len(cleaned), dim), dtype=np.float32)

    for idx, doc in enumerate(nlp.pipe(cleaned, batch_size=256)):
        valid_vecs = [
            tok.vector
            for tok in doc
            if tok.has_vector
            and (not exclude_stop or not tok.is_stop)
            and (not exclude_punct or not tok.is_punct)
        ]
        if valid_vecs:
            vectors[idx] = np.mean(valid_vecs, axis=0)
    return vectors


def concept_similarities(texts, concepts, prefix="sim", exclude_stop=True):
    """Return a DataFrame of cosine similarities between each text and each anchor concept."""
    s = pd.Series(texts)
    text_vecs = texts_to_vectors(s, exclude_stop=exclude_stop, exclude_punct=True)
    concept_vecs = texts_to_vectors(pd.Series(concepts), exclude_stop=exclude_stop, exclude_punct=True)
    sims = cosine_similarity(text_vecs, concept_vecs)
    cols = [f"{prefix}_{i}_{c}" for i, c in enumerate(concepts)]
    return pd.DataFrame(sims, index=s.index, columns=cols)


def discover_anchor_candidates(data, text_col, related_bool_cols=None, top_n=25):
    """Find top content lemmas in a text column and how often they appear when booleans are False."""
    nlp = get_nlp()
    sub = data[data[text_col].notna()].copy()
    if related_bool_cols:
        all_false = sub[related_bool_cols].fillna(False).astype(bool).sum(axis=1) == 0
    else:
        all_false = pd.Series(True, index=sub.index)

    total_counts = Counter()
    unaccounted_counts = Counter()

    for idx, doc in zip(sub.index, nlp.pipe(sub[text_col].astype(str), batch_size=256)):
        lemmas = {
            tok.lemma_.lower()
            for tok in doc
            if tok.has_vector and not tok.is_stop and not tok.is_punct and tok.is_alpha and len(tok.lemma_) > 2
        }
        total_counts.update(lemmas)
        if all_false.loc[idx]:
            unaccounted_counts.update(lemmas)

    rows = [
        {
            "candidate_word": word,
            "total_notes": tot,
            "notes_with_all_bools_false": unaccounted_counts[word],
        }
        for word, tot in total_counts.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def probe_anchor(data, text_col, anchor_phrase, top_n=12):
    """Inspect top-matching notes for a candidate anchor word or phrase."""
    non_empty = data[data[text_col].notna()].copy()
    sims = concept_similarities(non_empty[text_col], [anchor_phrase], prefix="probe")
    sim_col = sims.columns[0]
    ranked = non_empty.assign(**{sim_col: sims[sim_col]}).sort_values(sim_col, ascending=False)
    return ranked[[text_col, sim_col]].head(top_n)


class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    """Turn a single text column into 300-d spaCy document vectors."""

    def __init__(self, exclude_stop=True, exclude_punct=True):
        self.exclude_stop = exclude_stop
        self.exclude_punct = exclude_punct

    def fit(self, X, y=None):
        self.n_features_ = get_nlp().vocab.vectors_length
        return self

    def _texts(self, X):
        return X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X)

    def transform(self, X):
        s = self._texts(X)
        vecs = texts_to_vectors(s, exclude_stop=self.exclude_stop, exclude_punct=self.exclude_punct)
        return pd.DataFrame(vecs, index=s.index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        return np.array([f"vec_{i}" for i in range(self.n_features_)])


class ConceptSimilarityTransformer(BaseEstimator, TransformerMixin):
    """Turn a text column into cosine similarities against a list of anchor concepts."""

    def __init__(self, concepts=None, prefix="sim", exclude_stop=True, exclude_punct=True):
        self.concepts = concepts
        self.prefix = prefix
        self.exclude_stop = exclude_stop
        self.exclude_punct = exclude_punct

    def fit(self, X, y=None):
        self.concepts_ = list(self.concepts) if self.concepts is not None else []
        self.concept_vectors_ = texts_to_vectors(
            pd.Series(self.concepts_),
            exclude_stop=self.exclude_stop,
            exclude_punct=self.exclude_punct,
        )
        return self

    def _texts(self, X):
        return X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X)

    def transform(self, X):
        s = self._texts(X)
        text_vecs = texts_to_vectors(s, exclude_stop=self.exclude_stop, exclude_punct=self.exclude_punct)
        sims = cosine_similarity(text_vecs, self.concept_vectors_)
        return pd.DataFrame(sims, index=s.index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.prefix}_{i}_{c}" for i, c in enumerate(self.concepts_)])


def build_preprocessor(numeric_and_bool_cols, anchor_map):
    """Assemble a ColumnTransformer combining tabular columns and text anchor similarities."""
    transformers = [("tabular", "passthrough", numeric_and_bool_cols)]
    for col_name, (prefix, concepts) in anchor_map.items():
        if concepts:
            transformers.append(
                (f"{prefix}_sim", ConceptSimilarityTransformer(concepts=concepts, prefix=prefix), [col_name])
            )
    return ColumnTransformer(transformers, remainder="drop")