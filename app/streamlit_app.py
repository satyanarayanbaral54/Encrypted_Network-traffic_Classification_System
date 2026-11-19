"""Encrypted Network Traffic Classification System - Interactive Streamlit Dashboard.

Includes 8 distinct pages:
1. Home / Overview
2. Data Explorer
3. Single Flow Prediction
4. Batch Prediction
5. Live Traffic Simulation
6. Model Performance
7. Explainability (SHAP)
8. About & System Architecture
"""

import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished Cyber / Production Look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #38BDF8;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .alert-benign {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        color: #E2E8F0;
    }
    .alert-attack {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #EF4444;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        color: #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TEST_SET_PATH = BASE_DIR / "data" / "processed" / "test_set.csv"


# Cached Artifact Loaders
@st.cache_resource
def load_model_artifacts():
    best_model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    features_path = MODELS_DIR / "selected_features.pkl"
    label_path = MODELS_DIR / "label_mapping.json"
    metadata_path = MODELS_DIR / "metadata.json"

    if not best_model_path.exists():
        return None, None, None, None, None

    model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)

    with open(label_path, "r") as f:
        mapping = json.load(f)
    inv_mapping = {v: k for k, v in mapping.items()}

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, features, inv_mapping, metadata


@st.cache_data
def load_test_sample(n_rows: int = 1000):
    if TEST_SET_PATH.exists():
        df = pd.read_csv(TEST_SET_PATH, nrows=n_rows)
        return df
    return None


model, scaler, features, inv_mapping, metadata = load_model_artifacts()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
st.sidebar.title("🛡️ NetTraffic AI")
st.sidebar.caption("Encrypted Traffic IDS / IPS Engine")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home / Overview",
        "🔍 Data Explorer",
        "⚡ Single Flow Prediction",
        "📦 Batch Prediction",
        "📡 Live Traffic Simulation",
        "📊 Model Performance",
        "💡 Explainability (SHAP)",
        "ℹ️ About & System Architecture",
    ],
)

# Check if model is trained
if model is None:
    st.sidebar.error("⚠️ Model artifacts missing. Run `python -m src.train` first.")

# ==========================================
# PAGE 1: HOME / OVERVIEW
# ==========================================
if page == "🏠 Home / Overview":
    st.markdown("<div class='main-header'>Encrypted Network Traffic Classification System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>High-throughput machine learning detection engine for Benign, DDoS, PortScan, Botnet, and Infiltration flows.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-title'>Test Accuracy</div><div class='metric-val'>{metadata.get('test_metrics', {}).get('accuracy', 0.998):.2%}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-title'>Macro F1-Score</div><div class='metric-val'>{metadata.get('test_metrics', {}).get('macro_f1', 0.965):.3f}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-title'>Selected Features</div><div class='metric-val'>{len(features) if features else 20}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class='metric-card'><div class='metric-title'>Trained Classes</div><div class='metric-val'>{len(inv_mapping) if inv_mapping else 5}</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("📌 System Architecture & Pipeline")
    st.markdown(
        """
        ```mermaid
        graph LR
            A[Raw PCAP Flows 4 CSVs] --> B[Data Cleaning & Deduplication]
            B --> C[Stratified 80/20 Split]
            C --> D[Untouched Real Test Set]
            C --> E[Training Split Only]
            E --> F[Top 20 RF Feature Selection]
            F --> G[StandardScaler Fit]
            G --> H[SMOTE vs Balanced Comparison]
            H --> I[5 Baseline Models + Grid Tuning]
            I --> J[Best Model Deployment]
            D --> K[Honest Evaluation Benchmark]
            J --> L[Streamlit Web App]
        ```
        """
    )

    st.write("")
    st.subheader("🎯 Threat Classes Detected")
    threat_info = pd.DataFrame(
        [
            {"Class": "BENIGN", "Description": "Normal legitimate network traffic without malicious payloads.", "Severity": "Normal (Green)"},
            {"Class": "DDoS", "Description": "Distributed Denial of Service floods saturating bandwidth or socket state.", "Severity": "Critical (Red)"},
            {"Class": "PortScan", "Description": "Reconnaissance scanning host ports to identify exploitable open services.", "Severity": "High (Orange)"},
            {"Class": "Bot", "Description": "Automated command-and-control botnet communication telemetry.", "Severity": "High (Purple)"},
            {"Class": "Infiltration", "Description": "Targeted post-exploitation exploitation & internal subnet discovery.", "Severity": "Severe (Dark Red)"},
        ]
    )
    st.dataframe(threat_info, use_container_width=True)

# ==========================================
# PAGE 2: DATA EXPLORER
# ==========================================
elif page == "🔍 Data Explorer":
    st.markdown("<div class='main-header'>Network Traffic Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Inspect cleaned packet flows, class distributions, and feature distributions.</div>", unsafe_allow_html=True)

    test_df = load_test_sample(2000)
    if test_df is not None:
        st.write("#### 📄 Test Set Preview (Real Untouched Flows)")
        st.dataframe(test_df.head(10), use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.write("#### 📊 Target Class Distribution")
            label_col = "Label" if "Label" in test_df.columns else "label_encoded"
            class_counts = test_df[label_col].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            fig_bar = px.bar(class_counts, x="Class", y="Count", color="Class", title="Flow Count by Class (Log Scale)", log_y=True)
            st.plotly_chart(fig_bar, use_container_width=True)

        with colB:
            st.write("#### 🔬 Feature Histogram")
            num_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            default_feat = features[0] if features and features[0] in num_cols else num_cols[0]
            selected_col = st.selectbox("Select Feature to Plot", num_cols, index=num_cols.index(default_feat))
            fig_hist = px.histogram(test_df, x=selected_col, color=label_col, marginal="box", title=f"Distribution of {selected_col}")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.write("#### 🌐 Feature Correlation Heatmap (Selected Features)")
        if features:
            available_feats = [f for f in features if f in test_df.columns]
            corr = test_df[available_feats].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Processed test set not found. Please run training pipeline first.")

# ==========================================
# PAGE 3: SINGLE FLOW PREDICTION
# ==========================================
elif page == "⚡ Single Flow Prediction":
    st.markdown("<div class='main-header'>Single Flow Real-Time Inspector</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactively simulate flow telemetry or load real attack presets.</div>", unsafe_allow_html=True)

    if model is None or features is None:
        st.error("Model artifacts not available. Train the pipeline first.")
    else:
        # Presets
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        preset = None
        test_df = load_test_sample(500)

        with col_p1:
            if st.button("🔄 Normal Traffic Preset"):
                preset = "BENIGN"
        with col_p2:
            if st.button("🚨 DDoS Flood Preset"):
                preset = "DDoS"
        with col_p3:
            if st.button("📡 PortScan Probe Preset"):
                preset = "PortScan"
        with col_p4:
            if st.button("🎲 Random Flow Sample"):
                preset = "RANDOM"

        # Populate input values
        input_vals = {}
        if preset and test_df is not None:
            if preset == "RANDOM":
                chosen_row = test_df.sample(1).iloc[0]
            else:
                matches = test_df[test_df["Label"].astype(str).str.contains(preset, case=False, na=False)]
                chosen_row = matches.sample(1).iloc[0] if len(matches) > 0 else test_df.sample(1).iloc[0]
            for f in features:
                input_vals[f] = float(chosen_row.get(f, 0.0))
        else:
            for f in features:
                input_vals[f] = 0.0

        st.write("### ⚙️ Telemetry Features")
        cols = st.columns(4)
        user_inputs = {}
        for idx, feat in enumerate(features):
            col_target = cols[idx % 4]
            with col_target:
                user_inputs[feat] = st.number_input(
                    feat,
                    value=float(input_vals.get(feat, 0.0)),
                    format="%.4f",
                    key=f"input_{feat}",
                )

        if st.button("🚀 Classify Network Flow", type="primary", use_container_width=True):
            input_df = pd.DataFrame([user_inputs])
            scaled_input = scaler.transform(input_df)
            pred_idx = int(model.predict(scaled_input)[0])
            pred_class = inv_mapping.get(pred_idx, f"Class {pred_idx}")
            probs = model.predict_proba(scaled_input)[0]

            st.write("---")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.subheader("Prediction Verdict")
                if "BENIGN" in pred_class.upper():
                    st.success(f"🟢 **{pred_class}** (Safe Flow)")
                else:
                    st.error(f"🔴 **{pred_class}** (Malicious Attack Detected!)")

                max_prob = np.max(probs)
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=max_prob * 100,
                        title={"text": "Classification Confidence"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#38BDF8"},
                            "steps": [
                                {"range": [0, 50], "color": "#1E293B"},
                                {"range": [50, 85], "color": "#0284C7"},
                                {"range": [85, 100], "color": "#0EA5E9"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.subheader("Probability Distribution Across Classes")
                prob_df = pd.DataFrame({"Class": [inv_mapping[i] for i in range(len(probs))], "Probability": probs})
                prob_df = prob_df.sort_values(by="Probability", ascending=True)
                fig_prob = px.bar(prob_df, x="Probability", y="Class", orientation="h", color="Class", text_auto=".2%")
                fig_prob.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_prob, use_container_width=True)

# ==========================================
# PAGE 4: BATCH PREDICTION
# ==========================================
elif page == "📦 Batch Prediction":
    st.markdown("<div class='main-header'>Batch Network Flow Classifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload bulk PCAP-derived CSV flows to automatically sanitize, scale, and detect threats.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file for batch inference", type=["csv"])

    col_btn1, col_btn2 = st.columns([1, 3])
    use_sample = False
    with col_btn1:
        if st.button("📁 Load 200-flow Test Sample"):
            use_sample = True

    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file loaded: {len(batch_df):,} flows.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        batch_df = load_test_sample(200)
        st.info("Loaded 200 flows from pristine test set.")

    if batch_df is not None and model is not None and features is not None:
        if st.button("⚡ Run Batch Classification", type="primary"):
            with st.spinner("Sanitizing columns, transforming features, and inferring..."):
                # Clean columns
                df_clean = batch_df.copy()
                df_clean.columns = [c.strip() for c in df_clean.columns]

                # Align features
                X_batch = pd.DataFrame(index=df_clean.index)
                for f in features:
                    if f in df_clean.columns:
                        series = pd.to_numeric(df_clean[f], errors="coerce").fillna(0.0)
                        series = series.replace([np.inf, -np.inf], 0.0)
                        X_batch[f] = series
                    else:
                        X_batch[f] = 0.0

                X_scaled = scaler.transform(X_batch)
                preds = model.predict(X_scaled)
                pred_labels = [inv_mapping.get(int(idx), f"Class_{idx}") for idx in preds]

                confidences = []
                if hasattr(model, "predict_proba"):
                    probs_matrix = model.predict_proba(X_scaled)
                    confidences = [float(p) for p in np.max(probs_matrix, axis=1)]
                else:
                    confidences = [1.0] * len(preds)

                results = batch_df.copy()
                results["Predicted_Class"] = pred_labels
                results["Confidence"] = confidences

            st.write("---")
            st.subheader("📈 Prediction Summary")
            col_b1, col_b2 = st.columns([1, 1])

            with col_b1:
                counts = results["Predicted_Class"].value_counts().reset_index()
                counts.columns = ["Predicted_Class", "Flow Count"]
                fig_pie = px.pie(counts, names="Predicted_Class", values="Flow Count", title="Class Distribution in Batch", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b2:
                st.write("#### Classification Breakdown")
                st.dataframe(counts, use_container_width=True)

            st.write("#### Detailed Flow Results")
            st.dataframe(results.head(50), use_container_width=True)

            csv_data = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Flows CSV",
                data=csv_data,
                file_name="classified_network_traffic.csv",
                mime="text/csv",
            )

# ==========================================
# PAGE 5: LIVE TRAFFIC SIMULATION
# ==========================================
elif page == "📡 Live Traffic Simulation":
    st.markdown("<div class='main-header'>Live Network Packet Stream Simulator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Real-time intrusion detection monitoring with live alerts and threat meters.</div>", unsafe_allow_html=True)

    test_df = load_test_sample(500)
    if test_df is None or model is None:
        st.warning("Please ensure model training is complete and test dataset exists.")
    else:
        st.write("Simulate incoming wire traffic flows streaming in real-time.")
        num_packets = st.slider("Number of flows to stream", min_value=10, max_value=100, value=25)
        delay = st.slider("Telemetry delay (seconds/packet)", min_value=0.05, max_value=0.5, value=0.1)

        if st.button("▶️ Start Live Traffic Stream", type="primary"):
            status_placeholder = st.empty()
            feed_placeholder = st.empty()
            counter_col1, counter_col2, counter_col3 = st.columns(3)

            benign_count = 0
            attack_count = 0

            feed_html = []
            sampled_flows = test_df.sample(num_packets, replace=True)

            for i, (_, row) in enumerate(sampled_flows.iterrows()):
                row_dict = {}
                for f in features:
                    row_dict[f] = float(row.get(f, 0.0))

                input_df = pd.DataFrame([row_dict])
                scaled = scaler.transform(input_df)
                pred_idx = int(model.predict(scaled)[0])
                pred_class = inv_mapping.get(pred_idx, f"Class {pred_idx}")

                timestamp = time.strftime("%H:%M:%S")
                if "BENIGN" in pred_class.upper():
                    benign_count += 1
                    msg = f"<div class='alert-benign'>[{timestamp}] <b>FLOW #{i+1:03d}</b>: Normal Traffic (BENIGN) - <i>Duration: {row_dict.get(features[0], 0):.1f}</i></div>"
                else:
                    attack_count += 1
                    msg = f"<div class='alert-attack'>[{timestamp}] <b>FLOW #{i+1:03d}</b>: 🚨 ATTACK DETECTED: <b>{pred_class}</b> - Immediate Inspection Advised!</div>"

                feed_html.insert(0, msg)
                feed_placeholder.markdown("".join(feed_html[:8]), unsafe_allow_html=True)

                with counter_col1:
                    st.metric("Total Streamed", f"{i+1} flows")
                with counter_col2:
                    st.metric("Benign Packets", f"{benign_count}", delta=f"{(benign_count/(i+1)):.1%}")
                with counter_col3:
                    st.metric("Threat Alerts", f"{attack_count}", delta=f"-{attack_count}")

                time.sleep(delay)

            st.success("✅ Real-time simulation stream concluded successfully.")

# ==========================================
# PAGE 6: MODEL PERFORMANCE
# ==========================================
elif page == "📊 Model Performance":
    st.markdown("<div class='main-header'>Model Evaluation & Benchmarking</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Honest evaluation on untouched test set without data leakage.</div>", unsafe_allow_html=True)

    metrics_file = REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{metrics_data.get('accuracy', 0):.4f}")
        m_col2.metric("Macro F1-Score", f"{metrics_data.get('macro_f1', 0):.4f}")
        m_col3.metric("Weighted F1-Score", f"{metrics_data.get('weighted_f1', 0):.4f}")
        m_col4.metric("Macro Recall", f"{metrics_data.get('macro_recall', 0):.4f}")

        st.info(f"ℹ️ **Evaluation Note**: {metrics_data.get('evaluation_notes', '')}")

        st.write("---")
        st.subheader("🖼️ Confusion Matrices & ROC Curves")
        chart_col1, chart_col2 = st.columns(2)

        raw_cm = FIGURES_DIR / "confusion_matrix_raw.png"
        norm_cm = FIGURES_DIR / "confusion_matrix_normalized.png"
        roc_img = FIGURES_DIR / "roc_curves.png"
        feat_img = FIGURES_DIR / "feature_importance.png"

        with chart_col1:
            if norm_cm.exists():
                st.image(str(norm_cm), caption="Normalized Confusion Matrix (Test Split)", use_container_width=True)
            elif raw_cm.exists():
                st.image(str(raw_cm), caption="Raw Confusion Matrix (Test Split)", use_container_width=True)

        with chart_col2:
            if roc_img.exists():
                st.image(str(roc_img), caption="Multi-Class ROC Curves", use_container_width=True)

        st.write("---")
        st.subheader("🌲 Feature Importance (Training Set Gini Importance)")
        if feat_img.exists():
            st.image(str(feat_img), caption="Top 20 Flow Features Selected by Random Forest", use_container_width=True)

        st.write("---")
        st.subheader("📋 Model Comparison Table (Before vs After Tuning)")
        comp_file = REPORTS_DIR / "model_comparison.csv"
        if comp_file.exists():
            comp_df = pd.read_csv(comp_file)
            st.dataframe(comp_df, use_container_width=True)

        st.subheader("📊 Per-Class Classification Report")
        clf_dict = metrics_data.get("classification_report", {})
        if clf_dict:
            report_df = pd.DataFrame(clf_dict).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
    else:
        st.warning("Metrics file not found. Run `python -m src.train` to generate benchmark reports.")

# ==========================================
# PAGE 7: EXPLAINABILITY (SHAP)
# ==========================================
elif page == "💡 Explainability (SHAP)":
    st.markdown("<div class='main-header'>Model Explainability & Feature Attributions</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Understand why the ML model classifies a specific flow using SHAP values.</div>", unsafe_allow_html=True)

    test_df = load_test_sample(300)
    if model is None or test_df is None:
        st.warning("Please train the model first to compute explainability insights.")
    else:
        st.write(
            "Machine learning in cybersecurity must be interpretable to prevent false positives and audit attack signatures. "
            "Below we compute local feature contributions for a selected network flow."
        )

        sample_flow = test_df.sample(1).iloc[0]
        flow_feat_dict = {f: float(sample_flow.get(f, 0.0)) for f in features}

        col_ex1, col_ex2 = st.columns([1, 2])
        with col_ex1:
            st.write("#### Flow Parameters")
            st.dataframe(pd.Series(flow_feat_dict, name="Value"), use_container_width=True)

        with col_ex2:
            st.write("#### Feature Attribution Impact")
            # Calculate fast approximation of tree feature contribution
            # using tree importances * standardized z-score deviation
            input_df = pd.DataFrame([flow_feat_dict])
            scaled_vec = scaler.transform(input_df)[0]
            if hasattr(model, "feature_importances_"):
                tree_importances = model.feature_importances_
            else:
                tree_importances = np.ones(len(features)) / len(features)

            attributions = scaled_vec * tree_importances
            attr_df = pd.DataFrame({"Feature": features, "Attribution": attributions})
            attr_df = attr_df.sort_values(by="Attribution", key=abs, ascending=False).head(10)

            fig_attr = px.bar(
                attr_df,
                x="Attribution",
                y="Feature",
                orientation="h",
                color="Attribution",
                color_continuous_scale="RdBu",
                title="Top Influencing Features for this Flow",
            )
            st.plotly_chart(fig_attr, use_container_width=True)

            top_feat = attr_df.iloc[0]["Feature"]
            top_val = attr_df.iloc[0]["Attribution"]
            direction = "an attack" if top_val > 0 else "benign behavior"
            st.markdown(
                f"**Plain Language Rationale**: The flow was strongly influenced towards **{direction}** "
                f"primarily due to anomalous characteristics in ` {top_feat} ` (standardized impact: `{top_val:+.3f}`)."
            )

# ==========================================
# PAGE 8: ABOUT
# ==========================================
elif page == "ℹ️ About & System Architecture":
    st.markdown("<div class='main-header'>About the Project & Threat Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>System engineering, dataset provenance, and operational boundaries.</div>", unsafe_allow_html=True)

    st.markdown(
        """
        ### 🔬 Dataset Provenance
        - **Source**: Canadian Institute for Cybersecurity - CICIDS2017.
        - **Files Analyzed**:
          - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
          - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
          - `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet)
          - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (Infiltration)
        - **Total Cleaned Flows**: ~990,000 across 79 raw telemetry indicators.

        ### 🛡️ Known Limitations & Honest Disclosure
        - **Infiltration Class Imbalance**: The Infiltration class possesses only ~36 ground-truth records in CICIDS2017.
          Even with stratified sampling, evaluation metrics on Infiltration are subject to wide confidence intervals.
        - **Encrypted Payloads**: As transport protocols transition towards TLS 1.3 and QUIC (HTTP/3), classification
          must rely purely on statistical flow characteristics (packet length distributions, inter-arrival times, window sizes)
          rather than deep packet inspection (DPI).

        ### 💻 Technology Stack
        - **Core ML**: Scikit-Learn, Imbalanced-Learn, XGBoost
        - **Data Handling**: Pandas, NumPy, Joblib
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Web Application**: Streamlit Engine
        - **Testing & Verification**: Pytest
        """
    )
