"""
Custom sklearn-compatible transformers for the fraud-detection project.

This file is designed to support:
1. The imports used in Project (10).ipynb
2. The trained sklearn/imblearn pipeline
3. The Streamlit/SageMaker inference app

All transformers return pandas DataFrames when possible so that downstream
pipeline steps can keep column names.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _to_dataframe(X):
    """Convert input to a DataFrame without mutating the original object."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X).copy()


def _safe_divide(numerator, denominator):
    """Divide while avoiding inf values from zero denominators."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------
# General cleaning / preprocessing transformers
# ---------------------------------------------------------------------

class NumericImputer(BaseEstimator, TransformerMixin):
    """Fill numeric missing values with each column's median."""

    def __init__(self, strategy="median", fill_value=0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.numeric_cols_ = []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.numeric_cols_:
            if self.strategy == "mean":
                value = X[col].mean()
            elif self.strategy == "constant":
                value = self.fill_value
            else:
                value = X[col].median()

            if pd.isna(value):
                value = self.fill_value
            self.fill_values_[col] = value

        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col, value in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Fill categorical/object missing values with mode or a constant label."""

    def __init__(self, fill_value="missing"):
        self.fill_value = fill_value
        self.categorical_cols_ = []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.categorical_cols_ = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        for col in self.categorical_cols_:
            mode = X[col].dropna().mode()
            self.fill_values_[col] = mode.iloc[0] if len(mode) else self.fill_value

        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col, value in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        return X


class RecodeTextFeatures(BaseEstimator, TransformerMixin):
    """
    Clean text-like columns by standardizing case, whitespace, and common null labels.
    This is intentionally conservative so it does not destroy categorical meaning.
    """

    def __init__(self):
        self.text_cols_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.text_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        null_tokens = {"", "nan", "none", "null", "na", "n/a", "unknown"}

        for col in self.text_cols_:
            if col in X.columns:
                cleaned = X[col].astype(str).str.strip().str.lower()
                cleaned = cleaned.mask(cleaned.isin(null_tokens), "missing")
                X[col] = cleaned

        return X


class AdjustDataTypes(BaseEstimator, TransformerMixin):
    """
    Make light dtype fixes:
    - booleans become integers
    - object columns that are mostly numeric become numeric
    """

    def __init__(self, numeric_conversion_threshold=0.95):
        self.numeric_conversion_threshold = numeric_conversion_threshold
        self.convert_to_numeric_ = []
        self.bool_cols_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.bool_cols_ = X.select_dtypes(include=["bool"]).columns.tolist()
        self.convert_to_numeric_ = []

        obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in obj_cols:
            converted = pd.to_numeric(X[col], errors="coerce")
            ratio_numeric = converted.notna().mean()
            if ratio_numeric >= self.numeric_conversion_threshold:
                self.convert_to_numeric_.append(col)

        return self

    def transform(self, X):
        X = _to_dataframe(X)

        for col in self.bool_cols_:
            if col in X.columns:
                X[col] = X[col].astype(int)

        for col in self.convert_to_numeric_:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        return X


class CapOutliers(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns using fit-time quantile bounds."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.numeric_cols_ = []
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.numeric_cols_:
            lower = X[col].quantile(self.lower_quantile)
            upper = X[col].quantile(self.upper_quantile)
            if pd.isna(lower) or pd.isna(upper):
                lower, upper = X[col].min(), X[col].max()
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X


class ScaleNumericFeatures(BaseEstimator, TransformerMixin):
    """Standardize numeric columns while leaving categorical columns unchanged."""

    def __init__(self):
        self.numeric_cols_ = []
        self.means_ = {}
        self.stds_ = {}

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.numeric_cols_:
            mean = X[col].mean()
            std = X[col].std()
            self.means_[col] = 0 if pd.isna(mean) else mean
            self.stds_[col] = 1 if pd.isna(std) or std == 0 else std

        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.numeric_cols_:
            if col in X.columns:
                X[col] = (X[col] - self.means_[col]) / self.stds_[col]
        return X


# ---------------------------------------------------------------------
# Feature dropping transformers
# ---------------------------------------------------------------------

class DropHighMissing(BaseEstimator, TransformerMixin):
    """Drop columns with missing-value ratios above a threshold."""

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        missing_ratio = X.isna().mean()
        self.columns_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.columns_to_keep_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.columns_to_keep_]


class DropConstantFeatures(BaseEstimator, TransformerMixin):
    """Drop columns with only one unique value."""

    def __init__(self):
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        nunique = X.nunique(dropna=False)
        self.columns_to_keep_ = nunique[nunique > 1].index.tolist()
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.columns_to_keep_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.columns_to_keep_]


class DropNearConstantFeatures(BaseEstimator, TransformerMixin):
    """Drop columns where the most frequent value exceeds the given ratio."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        keep = []
        for col in X.columns:
            top_freq = X[col].value_counts(dropna=False, normalize=True).iloc[0]
            if top_freq < self.threshold:
                keep.append(col)
        self.columns_to_keep_ = keep
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.columns_to_keep_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.columns_to_keep_]


class DropHighCardinality(BaseEstimator, TransformerMixin):
    """Drop object/category columns with more than max_unique unique values."""

    def __init__(self, max_unique=200):
        self.max_unique = max_unique
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        keep = []
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                if X[col].nunique(dropna=False) <= self.max_unique:
                    keep.append(col)
            else:
                keep.append(col)
        self.columns_to_keep_ = keep
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.columns_to_keep_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.columns_to_keep_]


# ---------------------------------------------------------------------
# Fraud-specific feature engineering transformers
# ---------------------------------------------------------------------

class AddTimeFeatures(BaseEstimator, TransformerMixin):
    """Create interpretable time features from TransactionDT when available."""

    def __init__(self, transaction_col="TransactionDT"):
        self.transaction_col = transaction_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_dataframe(X)

        if self.transaction_col in X.columns:
            seconds = pd.to_numeric(X[self.transaction_col], errors="coerce")
            X["Transaction_hour"] = ((seconds // 3600) % 24).astype(float)
            X["Transaction_day"] = (seconds // (3600 * 24)).astype(float)
            X["Transaction_weekday"] = ((seconds // (3600 * 24)) % 7).astype(float)

        return X


class AddCardAddrInteraction(BaseEstimator, TransformerMixin):
    """Add card/address interaction features used commonly in fraud modeling."""

    def __init__(self):
        self.interaction_pairs = [
            ("card1", "addr1"),
            ("card1", "card2"),
            ("card1", "card3"),
            ("card4", "card6"),
            ("addr1", "addr2"),
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_dataframe(X)

        for left, right in self.interaction_pairs:
            if left in X.columns and right in X.columns:
                X[f"{left}_{right}_interaction"] = (
                    X[left].astype(str).fillna("missing") + "_" + X[right].astype(str).fillna("missing")
                )

        return X


class AddEmailMatch(BaseEstimator, TransformerMixin):
    """Create buyer/recipient email comparison features."""

    def __init__(self, payer_col="P_emaildomain", recipient_col="R_emaildomain"):
        self.payer_col = payer_col
        self.recipient_col = recipient_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_dataframe(X)

        if self.payer_col in X.columns and self.recipient_col in X.columns:
            payer = X[self.payer_col].astype(str).str.lower().fillna("missing")
            recip = X[self.recipient_col].astype(str).str.lower().fillna("missing")

            X["email_domain_match"] = (payer == recip).astype(int)
            X["payer_email_root"] = payer.str.split(".").str[0]
            X["recipient_email_root"] = recip.str.split(".").str[0]

        return X


class AddCardAvgAmount(BaseEstimator, TransformerMixin):
    """
    Add card-level TransactionAmt comparison features.
    During fit, it stores average transaction amount by selected card columns.
    """

    def __init__(self, amount_col="TransactionAmt", group_cols=None):
        self.amount_col = amount_col
        self.group_cols = group_cols
        self.global_mean_ = np.nan
        self.group_means_ = {}

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        if self.group_cols is None:
            self.group_cols_ = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in X.columns]
        else:
            self.group_cols_ = [c for c in self.group_cols if c in X.columns]

        if self.amount_col in X.columns:
            amount = pd.to_numeric(X[self.amount_col], errors="coerce")
            self.global_mean_ = amount.mean()
            if pd.isna(self.global_mean_):
                self.global_mean_ = 0

            for col in self.group_cols_:
                means = X.assign(_amount_=amount).groupby(col)["_amount_"].mean()
                self.group_means_[col] = means.to_dict()
        else:
            self.global_mean_ = 0

        return self

    def transform(self, X):
        X = _to_dataframe(X)

        if self.amount_col not in X.columns:
            return X

        amount = pd.to_numeric(X[self.amount_col], errors="coerce")

        for col in getattr(self, "group_cols_", []):
            if col in X.columns:
                mapped_mean = X[col].map(self.group_means_.get(col, {})).fillna(self.global_mean_)
                X[f"{col}_avg_{self.amount_col}"] = mapped_mean
                X[f"{col}_{self.amount_col}_ratio"] = _safe_divide(amount, mapped_mean)
                X[f"{col}_{self.amount_col}_diff"] = amount - mapped_mean

        return X


class AddAmountRatio(BaseEstimator, TransformerMixin):
    """Add log amount and ratios between TransactionAmt and selected numeric features."""

    def __init__(self, amount_col="TransactionAmt", ratio_prefixes=("C", "D")):
        self.amount_col = amount_col
        self.ratio_prefixes = ratio_prefixes
        self.ratio_cols_ = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.ratio_cols_ = [
            col for col in numeric_cols
            if col != self.amount_col and any(str(col).startswith(prefix) for prefix in self.ratio_prefixes)
        ]
        return self

    def transform(self, X):
        X = _to_dataframe(X)

        if self.amount_col in X.columns:
            amount = pd.to_numeric(X[self.amount_col], errors="coerce")
            X[f"log_amt_{self.amount_col}"] = np.log1p(amount.clip(lower=0))

            for col in self.ratio_cols_:
                if col in X.columns:
                    denom = pd.to_numeric(X[col], errors="coerce")
                    X[f"{self.amount_col}_to_{col}_ratio"] = _safe_divide(amount, denom)

        return X


# ---------------------------------------------------------------------
# Existing project classes retained from original file
# ---------------------------------------------------------------------

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method="yeo-johnson")

    def fit(self, X, y=None):
        X = _to_dataframe(X)

        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self

        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()

        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X = _to_dataframe(X)

        if self.skewed_cols:
            existing_cols = [c for c in self.skewed_cols if c in X.columns]
            if existing_cols:
                X[existing_cols] = self.pt.transform(X[existing_cols])
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        X = _to_dataframe(X)

        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        cat_cols = X_filtered.select_dtypes(exclude="number").columns
        cols_to_drop = []

        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / max(len(X_filtered), 1)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)

        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        numeric_X = X_filtered.select_dtypes(include="number")
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df["target"] = y
            correlations = temp_df.corr()["target"].abs().drop("target")
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        X = _to_dataframe(X)
        for col in self.features_to_keep:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.features_to_keep]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[5, 10, 20]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = _to_dataframe(X)
        data = X_df.squeeze()
        X_out = pd.DataFrame(index=X_df.index)

        for w in self.windows:
            X_out[f"EMA_{w}"] = data.ewm(span=w, min_periods=w).mean()

            M = data.diff(w - 1)
            N = data.shift(w - 1)
            X_out[f"ROC_{w}"] = (M / N) * 100

            X_out[f"MOM_{w}"] = data.diff(w)

            delta = data.diff()
            u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
            d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
            avg_gain = u.ewm(com=w - 1, adjust=False).mean()
            avg_loss = d.ewm(com=w - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            X_out[f"RSI_{w}"] = 100 - (100 / (1 + rs))

            X_out[f"MA_{w}"] = data.rolling(w, min_periods=w).mean()

        return X_out


class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=60):
        self.window = window
        self.last_beta_ = None
        self.last_alpha_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if len(X) < self.window:
            raise ValueError(f"Data length {len(X)} is less than window size {self.window}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Extractor must be fitted before calling transform.")

        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=["price_a", "price_b"])
        else:
            df = X.copy()
            df.columns = ["price_a", "price_b"]

        df[["spread", "beta"]] = self._compute_rolling_regression(df)
        df["z_score"] = self._calculate_z_score(df["spread"])
        df["spread_std"] = df["spread"].rolling(self.window).std()
        df["beta_stability"] = df["beta"].rolling(self.window).std()

        return df

    def _compute_rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas = np.full(len(df), np.nan)

        a_vals = df["price_a"].values
        b_vals = df["price_b"].values

        for i in range(self.window, len(df)):
            y = a_vals[i - self.window:i]
            x = b_vals[i - self.window:i]
            x_with_const = sm.add_constant(x)

            model = sm.OLS(y, x_with_const).fit()

            alpha, beta = model.params[0], model.params[1]
            betas[i] = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)

            self.last_alpha_, self.last_beta_ = alpha, beta

        return pd.DataFrame({"spread": spreads, "beta": betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window).mean()
        rolling_std = spread_series.rolling(self.window).std()
        return (spread_series - rolling_mean) / rolling_std


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Optional text vectorizer. Requires gensim only when this class is used."""

    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "Word2VecTransformer requires gensim. Install gensim or remove this step."
            ) from exc

        sentences = [str(row[0]).split() for row in X]
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )
        return self

    def transform(self, X):
        def get_mean_vector(text):
            words = str(text).split()
            vectors = [self.model.wv[w] for w in words if w in self.model.wv]
            if not vectors:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)

        return np.array([get_mean_vector(row[0]) for row in X])
