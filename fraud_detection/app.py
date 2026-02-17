from __future__ import annotations
 
import base64
import io
import time
import uuid
import threading
from pathlib import Path
from typing import Optional
 
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, abort
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
APP_DIR = Path(__file__).resolve().parent
 
MODEL_PATH = APP_DIR / "fraud_rf_model.pkl"
SCALER_PATH = APP_DIR / "amount_scaler.pkl"
 
app = Flask(__name__)
 
DOWNLOADS: dict[str, tuple[bytes, str, float]] = {}
DOWNLOAD_TTL_SECONDS = 10 * 60  # 10 minutes
 
 
def cleanup_downloads() -> None:
    now = time.time()
    expired = [k for k, (_, _, ts) in DOWNLOADS.items() if now - ts > DOWNLOAD_TTL_SECONDS]
    for k in expired:
        DOWNLOADS.pop(k, None)
 
 
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    return df
 
 
def align_to_model_features(df: pd.DataFrame, mdl) -> pd.DataFrame:
    feature_names = getattr(mdl, "feature_names_in_", None)
    if feature_names is not None:
        df = df[list(feature_names)]
    return df
 
 
def fig_to_base64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
 
 
def make_pie_chart(fraud: int, legit: int) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.pie(
        [fraud, legit],
        labels=["Fraud", "Legit"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff6b6b", "#6bd96b"],
    )
    ax.set_title("Fraud vs Legit")
    return fig_to_base64_png(fig)
 
 
def make_fraud_amount_hist(out_df: pd.DataFrame) -> Optional[str]:
    if "Amount" not in out_df.columns:
        return None
 
    fraud_rows = out_df[out_df["Prediction"] == 1]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
 
    if fraud_rows.empty:
        ax.text(0.5, 0.5, "No Fraud Found", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        ax.hist(fraud_rows["Amount"], bins=20, color="#8b5cf6", edgecolor="white", alpha=0.9)
        ax.set_title("Fraud by Amount Range")
        ax.set_xlabel("Amount")
        ax.set_ylabel("Count")
 
    return fig_to_base64_png(fig)
 
 
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_error: Optional[str] = None
 
model = None
scaler = None
 
 
def _load_artifacts() -> None:
    global model, scaler, _model_error
    try:
        m = joblib.load(MODEL_PATH)
        s = joblib.load(SCALER_PATH)
        with _model_lock:
            model = m
            scaler = s
        _model_ready.set()
        print("Model + scaler loaded.")
    except Exception as e:
        _model_error = repr(e)
        print("Artifact load failed:", _model_error)
 
 
threading.Thread(target=_load_artifacts, daemon=True).start()
 
 
@app.get("/")
def index():
    return render_template("index.html")
 
 
@app.post("/predict")
def predict():
    cleanup_downloads()
 
    if not _model_ready.is_set():
        msg = "Model is still loading. Please try again in a few seconds."
        if _model_error:
            msg = f"Model failed to load: {_model_error}"
        return render_template("index.html", error=msg)
 
    with _model_lock:
        mdl = model
        scl = scaler
 
    file = request.files.get("file")
    if not file or not file.filename:
        return render_template("index.html", error="Please choose a CSV file.")
 
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return render_template("index.html", error=f"Error reading CSV: {e}")
 
    df = prepare_features(df)
 
    try:
        df = align_to_model_features(df, mdl)
    except Exception as e:
        return render_template("index.html", error=f"CSV columns don't match model features: {e}")
 
    if "Amount" in df.columns:
        try:
            df.loc[:, "Amount"] = scl.transform(df[["Amount"]])
        except Exception as e:
            return render_template("index.html", error=f"Amount scaling failed: {e}")
 
    try:
        preds = mdl.predict(df)
    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {e}")
 
    out = df.copy()
    out["Prediction"] = preds
 
    fraud_count = int((preds == 1).sum())
    legit_count = int((preds == 0).sum())
 
    # Charts (base64)
    pie_b64 = make_pie_chart(fraud_count, legit_count)
    hist_b64 = make_fraud_amount_hist(out)
 
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    token = uuid.uuid4().hex
    download_name = f"fraud_results_{Path(file.filename).stem}.csv"
    DOWNLOADS[token] = (csv_bytes, download_name, time.time())
 
    return render_template(
        "index.html",
        filename=file.filename,
        total=len(out),
        fraud=fraud_count,
        legit=legit_count,
        token=token,
        pie_b64=pie_b64,
        hist_b64=hist_b64,
    )
 
 
@app.get("/download/<token>")
def download(token: str):
    cleanup_downloads()
 
    item = DOWNLOADS.get(token)
    if not item:
        abort(404)
 
    csv_bytes, download_name, _ = item
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name=download_name,
    )
 
 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)