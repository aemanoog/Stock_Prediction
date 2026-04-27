"""
Streamlit fraud-detection app for the IEEE-CIS transaction fraud project.

Expected project layout on Streamlit Cloud / SageMaker deployment repo:

project_root/
├── app.py                         # this file, or rename this file to app.py
├── src/
│   ├── Custom_Classes.py
│   └── feature_utils.py
├── Portfolio/
│   └── X_train.csv                # optional but recommended for sample rows/default values
└── .streamlit/secrets.toml         # AWS credentials + endpoint config

Required secrets.toml shape:
[aws_credentials]
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_SESSION_TOKEN = "..."          # omit or leave blank if not using temporary credentials
AWS_BUCKET = "..."
AWS_ENDPOINT = "fraud-classifier-pipeline-endpoint-auto-24"
AWS_REGION = "us-east-1"
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sagemaker
import shap
import streamlit as st
from imblearn.pipeline import Pipeline
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

warnings.simplefilter("ignore")

# -----------------------------------------------------------------------------
# Page setup and path configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="wide",
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "src").exists() else CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Importing these modules helps joblib unpickle the deployed pipeline locally.
# If a Streamlit environment is missing optional packages, the app can still call
# the SageMaker endpoint; only local SHAP explanations may be disabled.
try:
    import src.Custom_Classes  # noqa: F401
    import src.feature_utils  # noqa: F401
except Exception as import_error:  # pragma: no cover - Streamlit UI handles this
    CUSTOM_IMPORT_ERROR = import_error
else:
    CUSTOM_IMPORT_ERROR = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_INFO: Dict[str, Any] = {
    "pipeline_tar": "fine_tuned_gbm_pipeline.tar.gz",
    "pipeline_s3_prefix": "sklearn-pipeline-deployment",
    "explainer_file": "explainer_project.shap",
    "explainer_s3_prefix": "explainer",
    # These are editable inputs shown to the user. The remaining columns come
    # from a selected sample row in X_train.csv so the endpoint receives the full
    # feature set expected by the trained pipeline.
    "primary_inputs": [
        "TransactionAmt",
        "C1",
        "C5",
        "C7",
        "C10",
        "card1",
        "card2",
        "addr1",
        "dist1",
    ],
    "class_labels": {0: "Legitimate", 1: "Fraud"},
}

# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_boto_session(
    aws_id: str,
    aws_secret: str,
    aws_token: Optional[str],
    region_name: str,
) -> boto3.Session:
    kwargs = {
        "aws_access_key_id": aws_id,
        "aws_secret_access_key": aws_secret,
        "region_name": region_name,
    }
    if aws_token:
        kwargs["aws_session_token"] = aws_token
    return boto3.Session(**kwargs)


@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    """Load a small reference dataset for defaults/sample rows."""
    possible_paths = [
        PROJECT_ROOT / "Portfolio" / "X_train.csv",
        CURRENT_DIR / "Portfolio" / "X_train.csv",
        PROJECT_ROOT / "X_train.csv",
        CURRENT_DIR / "X_train.csv",
    ]

    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path, nrows=500)
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
            if "isFraud" in df.columns:
                df = df.drop(columns=["isFraud"])
            return df

    return pd.DataFrame()


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract a tar file into path."""
    target = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(target)):
            raise ValueError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(path=path)


@st.cache_resource(show_spinner=False)
def load_pipeline_from_s3(
    aws_id: str,
    aws_secret: str,
    aws_token: Optional[str],
    bucket: str,
    region_name: str,
    pipeline_tar: str,
    pipeline_s3_prefix: str,
) -> Any:
    """Download and load the trained sklearn/imblearn pipeline from S3."""
    session = get_boto_session(aws_id, aws_secret, aws_token, region_name)
    s3_client = session.client("s3")

    workdir = Path(tempfile.mkdtemp(prefix="fraud_pipeline_"))
    local_tar = workdir / pipeline_tar
    s3_key = f"{pipeline_s3_prefix}/{pipeline_tar}"
    s3_client.download_file(bucket, s3_key, str(local_tar))

    with tarfile.open(local_tar, "r:gz") as tar:
        _safe_extract(tar, workdir)
        joblib_names = [name for name in tar.getnames() if name.endswith(".joblib")]

    if not joblib_names:
        raise FileNotFoundError("No .joblib model file was found inside the model tar.gz file.")

    # Add extracted src folder to Python path so custom transformers can unpickle.
    extracted_src = workdir / "src"
    if extracted_src.exists() and str(workdir) not in sys.path:
        sys.path.insert(0, str(workdir))

    return joblib.load(workdir / joblib_names[0])


@st.cache_resource(show_spinner=False)
def load_explainer_from_s3(
    aws_id: str,
    aws_secret: str,
    aws_token: Optional[str],
    bucket: str,
    region_name: str,
    explainer_file: str,
    explainer_s3_prefix: str,
) -> Any:
    """Download and load the saved SHAP explainer from S3."""
    session = get_boto_session(aws_id, aws_secret, aws_token, region_name)
    s3_client = session.client("s3")

    local_path = Path(tempfile.gettempdir()) / explainer_file
    s3_key = f"{explainer_s3_prefix}/{explainer_file}"
    if not local_path.exists():
        s3_client.download_file(bucket, s3_key, str(local_path))

    # Your notebook saved the explainer with joblib.dump, so joblib.load is the
    # most reliable loading method for this project.
    return joblib.load(local_path)


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------
def get_aws_config() -> Tuple[str, str, Optional[str], str, str, str]:
    creds = st.secrets.get("aws_credentials", {})
    aws_id = creds.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = creds.get("AWS_SECRET_ACCESS_KEY", "")
    aws_token = creds.get("AWS_SESSION_TOKEN", "") or None
    bucket = creds.get("AWS_BUCKET", "")
    endpoint = creds.get("AWS_ENDPOINT", "")
    region = creds.get("AWS_REGION", "us-east-1")

    missing = [
        name
        for name, value in {
            "AWS_ACCESS_KEY_ID": aws_id,
            "AWS_SECRET_ACCESS_KEY": aws_secret,
            "AWS_BUCKET": bucket,
            "AWS_ENDPOINT": endpoint,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Streamlit secrets: {', '.join(missing)}")

    return aws_id, aws_secret, aws_token, bucket, endpoint, region


def coerce_prediction(raw_pred: Any) -> Tuple[int, Optional[float], Any]:
    """Handle common SageMaker response shapes: scalar, list, array, or probabilities."""
    arr = np.asarray(raw_pred)

    if arr.size == 0:
        raise ValueError("The endpoint returned an empty prediction.")

    # Examples handled:
    # [1]
    # [[1]]
    # [[0.12, 0.88]] probability vector
    # [0.88] fraud probability
    flat = arr.ravel()

    fraud_probability: Optional[float] = None
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        fraud_probability = float(arr.reshape(-1, 2)[0, 1])
        pred_class = int(fraud_probability >= 0.5)
    else:
        value = float(flat[-1])
        if 0.0 <= value <= 1.0 and not float(value).is_integer():
            fraud_probability = value
            pred_class = int(value >= 0.5)
        else:
            pred_class = int(round(value))

    return pred_class, fraud_probability, raw_pred


def call_sagemaker_endpoint(input_df: pd.DataFrame) -> Tuple[str, Optional[float], Any]:
    aws_id, aws_secret, aws_token, _bucket, endpoint, region = get_aws_config()
    boto_session = get_boto_session(aws_id, aws_secret, aws_token, region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    predictor = Predictor(
        endpoint_name=endpoint,
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer(),
    )

    payload = input_df.to_dict(orient="records")[0]
    raw_pred = predictor.predict(payload)
    pred_class, fraud_probability, _ = coerce_prediction(raw_pred)
    label = MODEL_INFO["class_labels"].get(pred_class, str(pred_class))
    return label, fraud_probability, raw_pred


def build_input_row(reference_df: pd.DataFrame, row_index: int, edited_values: Dict[str, Any]) -> pd.DataFrame:
    """Create a full one-row model input using a reference row plus user edits."""
    if reference_df.empty:
        row = pd.DataFrame([edited_values])
    else:
        row = reference_df.iloc[[row_index]].copy()
        for col, value in edited_values.items():
            if col in row.columns:
                row.loc[row.index[0], col] = value
            else:
                row[col] = value

    return row.reset_index(drop=True)


def numeric_bounds(series: pd.Series) -> Tuple[float, float, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 1.0, 0.0, 0.01

    min_val = float(clean.quantile(0.01))
    max_val = float(clean.quantile(0.99))
    default = float(clean.median())
    step = max((max_val - min_val) / 100, 0.01)

    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0

    return min_val, max_val, default, step


# -----------------------------------------------------------------------------
# Explainability helpers
# -----------------------------------------------------------------------------
def get_preprocessing_pipeline(best_pipeline: Any) -> Pipeline:
    """Return every step before SMOTE/sampler and model."""
    if not hasattr(best_pipeline, "steps"):
        raise TypeError("The loaded object does not look like a sklearn/imblearn pipeline.")

    steps = list(best_pipeline.steps)
    stop_names = {"sampler", "model"}
    preprocessing_steps = []
    for name, step in steps:
        if name in stop_names:
            break
        preprocessing_steps.append((name, step))

    if not preprocessing_steps:
        raise ValueError("No preprocessing steps were found before the model step.")

    return Pipeline(steps=preprocessing_steps)


def display_shap_explanation(input_df: pd.DataFrame) -> None:
    aws_id, aws_secret, aws_token, bucket, _endpoint, region = get_aws_config()

    if CUSTOM_IMPORT_ERROR is not None:
        st.warning(
            "Local SHAP explanation could not import the custom src modules. "
            "Prediction can still run through SageMaker."
        )
        st.caption(f"Import detail: {CUSTOM_IMPORT_ERROR}")
        return

    try:
        best_pipeline = load_pipeline_from_s3(
            aws_id,
            aws_secret,
            aws_token,
            bucket,
            region,
            MODEL_INFO["pipeline_tar"],
            MODEL_INFO["pipeline_s3_prefix"],
        )
        explainer = load_explainer_from_s3(
            aws_id,
            aws_secret,
            aws_token,
            bucket,
            region,
            MODEL_INFO["explainer_file"],
            MODEL_INFO["explainer_s3_prefix"],
        )
        preprocessing_pipeline = get_preprocessing_pipeline(best_pipeline)
        transformed = preprocessing_pipeline.transform(input_df)
        transformed_df = pd.DataFrame(transformed)
        shap_values = explainer(transformed_df, check_additivity=False)

        st.subheader("Decision Transparency")
        fig = plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)

        values = np.asarray(shap_values[0].values).ravel()
        feature_names = getattr(shap_values[0], "feature_names", None) or [
            f"feature_{i}" for i in range(len(values))
        ]
        top_idx = int(np.argmax(np.abs(values)))
        st.info(f"Most influential transformed feature: **{feature_names[top_idx]}**")
    except Exception as exc:
        st.warning("Prediction worked, but the local SHAP explanation could not be displayed.")
        st.caption(f"Explanation detail: {exc}")


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("💳 Fraud Detection Classifier")
st.write(
    "Use a sample transaction as the base record, edit the most important input "
    "fields, and send the full transaction row to the deployed SageMaker model."
)

with st.sidebar:
    st.header("App Settings")
    st.caption("The trained Gradient Boosting pipeline is called through your SageMaker endpoint.")
    show_raw_payload = st.checkbox("Show model input payload", value=False)
    show_raw_response = st.checkbox("Show raw endpoint response", value=False)
    enable_shap = st.checkbox("Show SHAP explanation", value=True)

reference_df = load_reference_data()

if reference_df.empty:
    st.warning(
        "No reference X_train.csv file was found. The app will only send the fields entered below. "
        "For the fraud pipeline, keep Portfolio/X_train.csv in the project so the model receives a full row."
    )
    row_index = 0
else:
    st.success(f"Loaded {len(reference_df):,} reference rows for default transaction values.")
    row_index = st.slider(
        "Choose a sample transaction row to use as the base input",
        min_value=0,
        max_value=max(len(reference_df) - 1, 0),
        value=0,
        step=1,
    )

available_inputs = [
    col for col in MODEL_INFO["primary_inputs"] if reference_df.empty or col in reference_df.columns
]

if not available_inputs and not reference_df.empty:
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    available_inputs = numeric_cols[:8]

st.subheader("Editable Transaction Inputs")

edited_values: Dict[str, Any] = {}
with st.form("prediction_form"):
    cols = st.columns(3)
    for i, col_name in enumerate(available_inputs):
        with cols[i % 3]:
            if reference_df.empty or col_name not in reference_df.columns:
                edited_values[col_name] = st.number_input(col_name, value=0.0, step=0.01)
                continue

            sample_value = reference_df.iloc[row_index][col_name]
            if pd.api.types.is_numeric_dtype(reference_df[col_name]):
                min_val, max_val, default, step = numeric_bounds(reference_df[col_name])
                current_value = sample_value if pd.notna(sample_value) else default
                edited_values[col_name] = st.number_input(
                    col_name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(current_value),
                    step=float(step),
                )
            else:
                options = reference_df[col_name].dropna().astype(str).unique().tolist()[:100]
                current_value = "" if pd.isna(sample_value) else str(sample_value)
                if current_value not in options:
                    options = [current_value] + options
                edited_values[col_name] = st.selectbox(col_name, options=options)

    submitted = st.form_submit_button("Run Fraud Prediction", type="primary")

input_df = build_input_row(reference_df, row_index, edited_values)

if show_raw_payload:
    st.subheader("Model Input Preview")
    st.dataframe(input_df, use_container_width=True)

if submitted:
    try:
        with st.spinner("Calling SageMaker endpoint..."):
            label, fraud_probability, raw_response = call_sagemaker_endpoint(input_df)

        result_col, prob_col = st.columns(2)
        with result_col:
            st.metric("Prediction", label)
        with prob_col:
            if fraud_probability is not None:
                st.metric("Estimated Fraud Probability", f"{fraud_probability:.2%}")
            else:
                st.metric("Estimated Fraud Probability", "Not returned")

        if label.lower() == "fraud":
            st.error("This transaction was classified as potentially fraudulent.")
        else:
            st.success("This transaction was classified as legitimate.")

        if show_raw_response:
            st.subheader("Raw Endpoint Response")
            st.write(raw_response)

        if enable_shap:
            display_shap_explanation(input_df)

    except Exception as exc:
        st.error("The prediction could not be completed.")
        st.exception(exc)

with st.expander("Deployment checklist"):
    st.markdown(
        """
        - Put this file at the project root and rename it to `app.py` if Streamlit Cloud expects that name.
        - Keep `src/Custom_Classes.py` and `src/feature_utils.py` in the repo so the saved pipeline can unpickle.
        - Keep `Portfolio/X_train.csv` in the repo if you want full-row sample defaults.
        - Add AWS keys, bucket, endpoint, and region to `.streamlit/secrets.toml`.
        - Confirm the SageMaker endpoint is running before using the app.
        """
    )
