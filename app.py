import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Intrusion Detection System Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# --- INLINE CUSTOM CSS: minimal cybersecurity theme ---
st.markdown("""
<style>
/* Basic color scheme for cybersecurity blue theme */
body, .main { 
    background-color: #001a2d !important; 
    color: #e8f6ff !important;
}

.stMarkdown, .stText, p, span {
    color: #e8f6ff !important;
}

h1, h2, h3 {
    color: #00ffff !important;
}

.stButton button {
    background-color: rgba(0,150,255,0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0,200,255,0.4) !important;
}

.stButton button:hover {
    background-color: rgba(0,150,255,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Project Controls")
# Removed theme selector - using cybersecurity blue theme only
uploaded_model = st.sidebar.file_uploader("Upload a trained model (.pkl or .joblib)", type=['pkl', 'joblib'])
uploaded_data = st.sidebar.file_uploader("Upload dataset (CSV)", type=['csv'])
st.sidebar.markdown("---")

if st.sidebar.button("Train Quick LogisticRegression Model"):
    st.session_state['train_quick'] = True

# --- BLUE THEME APPLIED (see CSS above) ---

# --- LOAD DATASET ---
data = None
if uploaded_data is not None:
    # If the user uploads a CSV, read it into a DataFrame
    data = pd.read_csv(uploaded_data)
    st.sidebar.success("✅ Dataset uploaded successfully.")
else:
    for filename in ["data.csv", "KDDTrain+.csv", "kdd.csv"]:
        if os.path.exists(filename):
            # Otherwise try to auto-load common filenames if they exist
            data = pd.read_csv(filename)
            st.sidebar.info(f"Loaded dataset: {filename}")
            break

# --- LOAD MODEL ---
model = None
if uploaded_model is not None:
    try:
        # Load a serialized model provided by the user (pickle/joblib)
        model = joblib.load(uploaded_model)
        st.sidebar.success("✅ Model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")

if model is None and os.path.exists("trained_model.pkl"):
    try:
        model = joblib.load("trained_model.pkl")
        st.sidebar.info("Loaded saved model: trained_model.pkl")
    except Exception as e:
        st.sidebar.warning("Could not load saved model: " + str(e))

# --- QUICK TRAIN MODEL ---
if 'train_quick' in st.session_state and st.session_state['train_quick']:
    if data is None:
        st.error("Please upload a dataset first.")
    else:
        st.info("Training LogisticRegression model...")
        with st.spinner("Training in progress..."):
            df = data.copy()
            # Try to auto-detect common target/label column names
            target_candidates = [c for c in df.columns if c.lower() in ('label','attack','target','class')]
            if not target_candidates:
                st.error("No valid target column found. Add a column named 'label', 'attack', or 'class'.")
            else:
                target = target_candidates[0]
                # Use only numeric features for quick training; fill missing values
                X = df.drop(columns=[target]).select_dtypes(include=[np.number])
                y = df[target]
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X.fillna(0))
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                clf = LogisticRegression(max_iter=1000)
                start = time.time()
                clf.fit(X_train, y_train)
                end = time.time()
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                joblib.dump({'model': clf, 'scaler': scaler, 'feature_columns': list(X.columns)}, "trained_model.pkl")
                st.success(f"Model trained successfully! ✅ Accuracy: {acc*100:.2f}% (Training time: {end-start:.2f}s)")
                model = {'model': clf, 'scaler': scaler, 'feature_columns': list(X.columns)}

# --- VALIDATE MODEL FORMAT ---
if model is not None and not isinstance(model, dict):
    model = {'model': model, 'scaler': None, 'feature_columns': None}

# --- MAIN DASHBOARD ---
st.markdown("""
<div class="hero">
    <h1>🚨 Intrusion Detection System Dashboard</h1>
    <p>AI-powered, animated, and interactive security analytics UI for your final year project.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# --- LEFT COLUMN: DATA & PREDICTIONS ---
with col1:
    st.subheader("📊 Data Overview & Live Prediction")
    if data is not None:
        st.markdown(f"**Dataset Preview** — {data.shape[0]} rows × {data.shape[1]} columns")
        st.dataframe(data.head())

    uploaded_pred = st.file_uploader("Upload CSV for Batch Prediction", type=['csv'], key="predcsv")
    if uploaded_pred is not None:
        dfpred = pd.read_csv(uploaded_pred)
        if model is None:
            st.error("Please upload or train a model first.")
        else:
            cols = model.get('feature_columns')
            if cols is None:
                st.warning("Feature columns missing — using numeric columns only.")
                Xpred = dfpred.select_dtypes(include=[np.number]).fillna(0)
            else:
                missing = [c for c in cols if c not in dfpred.columns]
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                    Xpred = None
                else:
                    Xpred = dfpred[cols].fillna(0)
            if Xpred is not None:
                scaler = model.get('scaler')
                Xpred_scaled = scaler.transform(Xpred) if scaler else Xpred.values
                preds = model['model'].predict(Xpred_scaled)
                dfpred['Prediction'] = preds
                st.success("✅ Prediction complete!")
                
                # Calculate attack statistics for infographic
                attack_count = sum(1 for p in preds if 'attack' in str(p).lower())
                normal_count = len(preds) - attack_count
                attack_pct = (attack_count / len(preds) * 100) if len(preds) > 0 else 0
                
                # Display prediction statistics as infographic cards
                st.subheader("📊 Prediction Summary")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("🚨 Attacks Detected", f"{attack_count}", delta=f"{attack_pct:.1f}%")
                with col_stats2:
                    st.metric("✅ Normal Traffic", f"{normal_count}", delta=f"{100-attack_pct:.1f}%")
                with col_stats3:
                    st.metric("📈 Total Samples", len(preds))
                
                # Add a visual progress bar
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #ff4444 {attack_pct}%, #22c55e {attack_pct}%);
                            height: 20px; border-radius: 10px; margin: 10px 0;"></div>
                <p style="text-align: center; font-weight: bold; font-size: 14px;">
                    {attack_pct:.1f}% Attack | {100-attack_pct:.1f}% Normal
                </p>
                """, unsafe_allow_html=True)
                
                # Add graphs section
                st.subheader("📈 Prediction Analytics & Visualizations")
                
                # Create visualization columns
                graph_col1, graph_col2 = st.columns(2)
                
                with graph_col1:
                    # Pie chart for attack vs normal
                    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                    colors = ['#ff4444', '#22c55e']
                    wedges, texts, autotexts = ax_pie.pie(
                        [attack_count, normal_count],
                        labels=['🚨 Attack', '✅ Normal'],
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90,
                        textprops={'fontsize': 11, 'weight': 'bold'}
                    )
                    ax_pie.set_title('Attack Distribution', fontsize=12, fontweight='bold', pad=20)
                    st.pyplot(fig_pie)
                
                with graph_col2:
                    # Bar chart for attack vs normal counts
                    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                    categories = ['Attack', 'Normal']
                    counts = [attack_count, normal_count]
                    bars = ax_bar.bar(categories, counts, color=['#ff4444', '#22c55e'], edgecolor='black', linewidth=1.5)
                    ax_bar.set_ylabel('Count', fontsize=11, fontweight='bold')
                    ax_bar.set_title('Traffic Classification', fontsize=12, fontweight='bold', pad=20)
                    ax_bar.set_ylim(0, max(counts) * 1.1)
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontweight='bold')
                    st.pyplot(fig_bar)
                
                # Service distribution analysis (if 'service' column exists)
                if 'service' in dfpred.columns:
                    st.subheader("🔧 Service Distribution Analysis")
                    service_counts = dfpred['service'].value_counts()
                    
                    fig_service, ax_service = plt.subplots(figsize=(8, 4))
                    ax_service.barh(service_counts.index, service_counts.values, color='#0096c7', edgecolor='black', linewidth=1.2)
                    ax_service.set_xlabel('Count', fontsize=11, fontweight='bold')
                    ax_service.set_title('Network Services Used', fontsize=12, fontweight='bold', pad=20)
                    # Add value labels
                    for i, v in enumerate(service_counts.values):
                        ax_service.text(v, i, f' {v}', va='center', fontweight='bold')
                    st.pyplot(fig_service)
                
                # Duration vs Destination Bytes (Attack vs Normal)
                if 'duration' in dfpred.columns and 'dst_bytes' in dfpred.columns:
                    st.subheader("🔍 Feature Correlation Analysis")
                    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
                    
                    # Separate attack and normal predictions
                    attack_mask = dfpred['Prediction'].str.lower().str.contains('attack', na=False)
                    
                    ax_scatter.scatter(dfpred[~attack_mask]['duration'], 
                                      dfpred[~attack_mask]['dst_bytes'],
                                      c='#22c55e', label='Normal', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                    ax_scatter.scatter(dfpred[attack_mask]['duration'],
                                      dfpred[attack_mask]['dst_bytes'],
                                      c='#ff4444', label='Attack', alpha=0.8, s=80, edgecolors='darkred', linewidth=0.5, marker='^')
                    
                    ax_scatter.set_xlabel('Duration (seconds)', fontsize=11, fontweight='bold')
                    ax_scatter.set_ylabel('Destination Bytes', fontsize=11, fontweight='bold')
                    ax_scatter.set_title('Duration vs Destination Bytes', fontsize=12, fontweight='bold', pad=20)
                    ax_scatter.legend(loc='upper right', fontsize=10)
                    ax_scatter.grid(True, alpha=0.3)
                    st.pyplot(fig_scatter)
                
                # Display full prediction results with enhanced styling
                st.subheader("📋 Detailed Prediction Results")
                
                # Enhanced style function with badges
                def style_predictions(row):
                    pred = str(row['Prediction']).lower()
                    # Highlight attack rows with red background and white bold text
                    if 'attack' in pred:
                        return ['background-color: #ff4444; color: white; font-weight: bold; text-align: center;'] * len(row)
                    else:
                        return ['background-color: #22c55e; color: white; font-weight: bold; text-align: center;'] * len(row)
                
                styled_df = dfpred.style.apply(style_predictions, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Additional statistics section
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    avg_duration = dfpred['duration'].mean() if 'duration' in dfpred.columns else 0
                    st.metric("⏱️ Avg Duration (s)", f"{avg_duration:.2f}")
                
                with stats_col2:
                    avg_src_bytes = dfpred['src_bytes'].mean() if 'src_bytes' in dfpred.columns else 0
                    st.metric("📤 Avg Source Bytes", f"{int(avg_src_bytes)}")
                
                with stats_col3:
                    avg_dst_bytes = dfpred['dst_bytes'].mean() if 'dst_bytes' in dfpred.columns else 0
                    st.metric("📥 Avg Dest Bytes", f"{int(avg_dst_bytes)}")
                
                with stats_col4:
                    threat_level = "🔴 HIGH" if attack_pct > 50 else "🟡 MEDIUM" if attack_pct > 20 else "🟢 LOW"
                    st.metric("⚠️ Threat Level", threat_level)
                
                # Download option
                csv_bytes = dfpred.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

# --- RIGHT COLUMN: MODEL METRICS ---
with col2:
    st.subheader("🧠 Model & Performance Metrics")
    if model is None:
        st.info("Upload or train a model to view metrics.")
    else:
        st.markdown("**Model Type:**")
        st.info(f"📦 {type(model['model']).__name__}")

        if data is not None and model.get('feature_columns') is not None:
            df = data.copy()
            target_candidates = [c for c in df.columns if c.lower() in ('label','attack','target','class')]
            if target_candidates:
                target = target_candidates[0]
                X = df[model['feature_columns']].fillna(0)
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                X_scaled = model['scaler'].transform(X_test) if model.get('scaler') else X_test
                preds = model['model'].predict(X_scaled)
                
                # Calculate accuracy
                acc = accuracy_score(y_test, preds)
                st.metric("Accuracy Score", f"{acc*100:.2f}%")
                
                # Confusion matrix visualization
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(cm, cmap="RdYlGn")
                ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontweight='bold')
                st.pyplot(fig)
                
                # Classification metrics summary
                st.markdown("**Classification Report:**")
                report_dict = classification_report(y_test, preds, output_dict=True)
                
                # Display as metrics in columns for visual appeal
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    precision = report_dict.get('weighted avg', {}).get('precision', 0)
                    st.metric("📌 Precision", f"{precision*100:.2f}%")
                with metric_cols[1]:
                    recall = report_dict.get('weighted avg', {}).get('recall', 0)
                    st.metric("🎯 Recall", f"{recall*100:.2f}%")
                
                # Show classification report text
                st.text(classification_report(y_test, preds))

# --- FOOTER ---
st.markdown("""
<div class="footer">
  <div class="pulse"></div>
  <p>Intrusion Detection Dashboard — built with ❤️ using Streamlit.</p>
</div>
""", unsafe_allow_html=True)
