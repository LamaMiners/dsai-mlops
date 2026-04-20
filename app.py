import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy as np
import pickle
import os

# ─────────────────────────────
# Config
# ─────────────────────────────
st.set_page_config(page_title="Phishing Detection", layout="wide")
st.title("Phishing Detection Dashboard")

# ─────────────────────────────
# Model laden
# ─────────────────────────────
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.sklearn.load_model(
        model_uri="models:/PhishingDetectionBestModel/2"  # Version 2 = dein fixes Modell
    )
    return model

model = load_model()

@st.cache_resource
def load_ols_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = mlflow.MlflowClient()
    
    runs = client.search_runs(
        experiment_ids=["1"],
        filter_string="tags.Algorithm = 'OLS'",
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )
    
    if not runs:
        return None
        
    run_id = runs[0].info.run_id
    
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="ols_regression_model"
    )
    
    import os
    import pickle
    model_file = os.path.join(artifact_path, "model.statsmodels")
    with open(model_file, "rb") as f:
        ols = pickle.load(f)
    
    return ols

ols_model = load_ols_model()

# Den Wert aus MLflow-Run
THRESHOLD = 0.9302 

# ─────────────────────────────
# Sidebar – Eingabe
# ─────────────────────────────
st.sidebar.header("Email / Text Features")
num_words             = st.sidebar.slider("Number of Words",           0, 1000, 100)
num_unique_words      = st.sidebar.slider("Unique Words",              0, 1000,  80)
num_stopwords         = st.sidebar.slider("Stopwords",                 0,  500,  30)
num_links             = st.sidebar.slider("Number of Links",           0,   50,   2)
num_unique_domains    = st.sidebar.slider("Unique Domains",            0,   20,   1)
num_email_addresses   = st.sidebar.slider("Email Addresses",           0,   20,   1)
num_spelling_errors   = st.sidebar.slider("Spelling Errors",           0,   50,   2)
num_urgent_keywords   = st.sidebar.slider("Urgent Keywords",           0,   20,   1)

input_df = pd.DataFrame({
    "num_words":            [num_words],
    "num_unique_words":     [num_unique_words],
    "num_stopwords":        [num_stopwords],
    "num_links":            [num_links],
    "num_unique_domains":   [num_unique_domains],
    "num_email_addresses":  [num_email_addresses],
    "num_spelling_errors":  [num_spelling_errors],
    "num_urgent_keywords":  [num_urgent_keywords],
})

# ─────────────────────────────
# Vorhersage
# ─────────────────────────────
st.header("Prediction")

# Live-Wahrscheinlichkeit immer anzeigen (ohne Button)
probability = model.predict_proba(input_df)[0][1]
st.metric("Phishing Probability", f"{probability:.2%}")
st.progress(float(probability))

if st.button("Run Analysis"):
    # ── Hauptvorhersage via Decision Tree + Threshold ──────────────────
    prediction = int(probability >= THRESHOLD)

    if prediction == 1:
        st.error(f"**Phishing detected!**  \nProbability: {probability:.2%}  \n(Threshold: {THRESHOLD:.2f})")
    else:
        st.success(f"**Legitimate email**  \nPhishing probability: {probability:.2%}  \n(Threshold: {THRESHOLD:.2f})")

    # ── Konfidenzintervall via OLS ─────────────────────────────────────
    st.subheader("Confidence Interval (OLS)")
    try:
        input_with_const = sm.add_constant(input_df, has_constant='add')
        pred_summary = ols_model.get_prediction(input_with_const).summary_frame(alpha=0.05)

        ci_mean  = pred_summary["mean"].values[0]
        ci_lower = pred_summary["mean_ci_lower"].values[0]
        ci_upper = pred_summary["mean_ci_upper"].values[0]

        # Werte auf [0,1] clippen – OLS kann theoretisch >1 oder <0 ausgeben
        ci_mean  = float(np.clip(ci_mean,  0, 1))
        ci_lower = float(np.clip(ci_lower, 0, 1))
        ci_upper = float(np.clip(ci_upper, 0, 1))

        st.info(
            f"**Predicted Phishing Probability:** {ci_mean:.2%}  \n"
            f"**95% Confidence Interval:** {ci_lower:.2%} – {ci_upper:.2%}"
        )

        # Visuelles KI-Intervall
        fig_ci, ax_ci = plt.subplots(figsize=(6, 1.8))
        ax_ci.barh(
            ["OLS Estimate"],
            [ci_upper - ci_lower],
            left=[ci_lower],
            color="#f39c12", alpha=0.5, label="95% CI"
        )
        ax_ci.plot(
            [ci_mean], [0],
            marker="D", color="#e74c3c", markersize=10, label="Point Estimate"
        )
        ax_ci.axvline(x=THRESHOLD, color="gray", linestyle="--", label=f"Threshold ({THRESHOLD})")
        ax_ci.set_xlim(0, 1)
        ax_ci.set_xlabel("Phishing Probability")
        ax_ci.set_title("95% Confidence Interval – OLS Regression")
        ax_ci.legend(loc="upper left", fontsize=8)
        st.pyplot(fig_ci)

        with st.expander("Explanation?"):
            st.write(f"""
            The OLS model estimates a phishing probability of **{ci_mean:.2%}**.  
            With 95% confidence, the true probability lies between **{ci_lower:.2%}** and **{ci_upper:.2%}**.  
            The dashed line marks the decision threshold of **{THRESHOLD}**.  
            If the entire CI is above the threshold → high confidence it's phishing.  
            If the CI crosses the threshold → borderline case, treat with caution.
            """)

    except Exception as e:
        st.warning(f"Confidence interval could not be calculated: {e}")

# ─────────────────────────────
# Key Insights
# ─────────────────────────────
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    # Feature Importance – funktioniert für Decision Tree UND Logistic Regression
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        title = "Feature Importance (Decision Tree)"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        title = "Feature Importance (Logistic Regression – abs. coefficients)"
    else:
        importance = None

    if importance is not None:
        features = input_df.columns
        sorted_idx = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(
            [features[i] for i in sorted_idx],
            importance[sorted_idx],
            color=["#e74c3c" if importance[i] > np.mean(importance) else "#3498db"
                   for i in sorted_idx]
        )
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)
    else:
        st.info("Feature Importance not available for this model type.")

with col2:
    # Gauge-Chart: aktuelle Eingabe visualisiert
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    values = input_df.iloc[0].values
    max_vals = [1000, 1000, 500, 50, 20, 20, 50, 20]
    normalized = [v/m for v, m in zip(values, max_vals)]
    colors = ["#e74c3c" if n > 0.5 else "#2ecc71" for n in normalized]
    ax2.barh(input_df.columns, normalized, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_title("Current Input – Normalized (red = high risk indicator)")
    ax2.set_xlabel("Relative Value (0 = min, 1 = max)")
    st.pyplot(fig2)

# ─────────────────────────────
# Threshold-Info
# ─────────────────────────────
st.divider()
st.caption(f"Model: PhishingDetectionBestModel v2 | Threshold: {THRESHOLD} | "
           f"Tracking URI: http://localhost:5000")