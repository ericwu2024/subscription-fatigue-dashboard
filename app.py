"""
Subscription Fatigue Prediction Dashboard
Early Warning Signals for the Streaming Economy
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Subscription Fatigue Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #2C3E50, #34495E);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .risk-high { border-left: 6px solid #E74C3C; }
    .risk-low { border-left: 6px solid #27AE60; }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
@st.cache_data
def load_data():
    raw = pd.read_csv('data/dataset_16_companies_with_yoy_diff.csv')
    reg_data = pd.read_csv('data/full_regression_data.csv')
    cls_data = pd.read_csv('data/full_classification_data.csv')
    reg_preds = pd.read_csv('data/regression_predictions.csv')
    cls_preds = pd.read_csv('data/classification_predictions.csv')
    with open('models/model_metadata.json') as f:
        metadata = json.load(f)
    return raw, reg_data, cls_data, reg_preds, cls_preds, metadata

@st.cache_resource
def load_models():
    reg_model = joblib.load('models/regression_model.joblib')
    reg_scaler = joblib.load('models/regression_scaler.joblib')
    cls_model = joblib.load('models/classification_model.joblib')
    cls_scaler = joblib.load('models/classification_scaler.joblib')
    return reg_model, reg_scaler, cls_model, cls_scaler

raw, reg_data, cls_data, reg_preds, cls_preds, metadata = load_data()
reg_model, reg_scaler, cls_model, cls_scaler = load_models()

companies = sorted(raw['Company'].unique())
industries = sorted(raw['Industry'].unique())

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Market Overview", "🔍 Company Deep Dive", "🧪 What-If Simulator", "📈 Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Subscription Fatigue Predictor** uses external signals 
    (Google Trends, Reddit, Macro data) to predict subscriber 
    growth trends — no internal data needed.
    """)
    st.markdown("---")
    st.markdown("**Team:** Eric Wu, Mala Ramakrishnan, Mina Mafi, Samuel Dominguez")
    st.markdown("*MIDS W210 Capstone*")

# ============================================================
# PAGE 1: MARKET OVERVIEW
# ============================================================
if page == "🏠 Market Overview":
    st.markdown('<p class="main-header">Subscription Fatigue Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Early Warning Signals for the Streaming Economy</p>', unsafe_allow_html=True)
    
    # Latest quarter risk scores
    latest_cls = cls_data[cls_data['Year'] == 2025].copy()
    if len(latest_cls) > 0:
        latest_q = latest_cls['Quarter'].max()
        latest = latest_cls[latest_cls['Quarter'] == latest_q].copy()
    else:
        latest = cls_data.groupby('Company').last().reset_index()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies Tracked", f"{len(companies)}")
    with col2:
        n_at_risk = (latest['Decline_Probability'] > 0.5).sum() if 'Decline_Probability' in latest.columns else 0
        st.metric("At Risk (>50%)", f"{n_at_risk}", delta=None)
    with col3:
        avg_prob = latest['Decline_Probability'].mean() * 100 if 'Decline_Probability' in latest.columns else 0
        st.metric("Avg Risk Score", f"{avg_prob:.1f}%")
    with col4:
        st.metric("Data Quarters", "20 (Q1 2020 - Q4 2024)")
    
    st.markdown("---")
    
    # Risk Heatmap
    st.subheader("🔥 Deceleration Risk Heatmap (Latest Quarter)")
    
    if 'Decline_Probability' in latest.columns:
        heatmap_data = latest[['Company', 'Industry', 'Decline_Probability']].copy()
        heatmap_data['Risk_Pct'] = (heatmap_data['Decline_Probability'] * 100).round(1)
        heatmap_data = heatmap_data.sort_values('Decline_Probability', ascending=False)
        
        fig = px.bar(
            heatmap_data,
            x='Company',
            y='Risk_Pct',
            color='Risk_Pct',
            color_continuous_scale=['#27AE60', '#F39C12', '#E74C3C'],
            range_color=[0, 100],
            labels={'Risk_Pct': 'Risk Score (%)'},
            text='Risk_Pct'
        )
        fig.update_layout(
            height=450,
            xaxis_tickangle=-45,
            yaxis_title="Deceleration Risk (%)",
            xaxis_title="",
            coloraxis_colorbar_title="Risk %",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Risk / Safe
    st.markdown("---")
    col_risk, col_safe = st.columns(2)
    
    with col_risk:
        st.subheader("⚠️ Highest Risk Companies")
        top_risk = heatmap_data.nlargest(5, 'Decline_Probability')[['Company', 'Industry', 'Risk_Pct']]
        top_risk.columns = ['Company', 'Industry', 'Risk Score (%)']
        st.dataframe(top_risk, use_container_width=True, hide_index=True)
    
    with col_safe:
        st.subheader("✅ Lowest Risk Companies")
        top_safe = heatmap_data.nsmallest(5, 'Decline_Probability')[['Company', 'Industry', 'Risk_Pct']]
        top_safe.columns = ['Company', 'Industry', 'Risk Score (%)']
        st.dataframe(top_safe, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 2: COMPANY DEEP DIVE
# ============================================================
elif page == "🔍 Company Deep Dive":
    st.markdown('<p class="main-header">Company Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore individual company risk profiles and growth trends</p>', unsafe_allow_html=True)
    
    selected_company = st.selectbox("Select a Company", companies, index=companies.index('Netflix'))
    
    # Filter data for selected company
    company_raw = raw[raw['Company'] == selected_company].sort_values(['Year', 'Quarter'])
    company_reg = reg_data[reg_data['Company'] == selected_company].sort_values(['Year', 'Quarter'])
    company_cls = cls_data[cls_data['Company'] == selected_company].sort_values(['Year', 'Quarter'])
    
    industry = company_raw['Industry'].iloc[0] if len(company_raw) > 0 else "Unknown"
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_sub = company_raw['Subscribers_Millions'].dropna().iloc[-1] if len(company_raw) > 0 else 0
    latest_growth = company_reg['Subscribers_Millions_yoy_growth_rate'].iloc[-1] * 100 if len(company_reg) > 0 else 0
    latest_risk = company_cls['Decline_Probability'].iloc[-1] * 100 if len(company_cls) > 0 and 'Decline_Probability' in company_cls.columns else 0
    
    with col1:
        st.metric("Industry", industry)
    with col2:
        st.metric("Latest Subscribers", f"{latest_sub:.1f}M")
    with col3:
        st.metric("YoY Growth Rate", f"{latest_growth:.1f}%", delta=f"{latest_growth:.1f}%")
    with col4:
        risk_color = "🔴" if latest_risk > 60 else "🟡" if latest_risk > 40 else "🟢"
        st.metric("Deceleration Risk", f"{risk_color} {latest_risk:.1f}%")
    
    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Growth Trends", "🎯 Risk Timeline", "🔬 Driver Analysis"])
    
    with tab1:
        # Subscriber count + YoY growth
        if len(company_raw) > 0:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Subscriber Count (Millions)", "YoY Growth Rate"),
                              vertical_spacing=0.12)
            
            fig.add_trace(
                go.Scatter(x=company_raw['Quarter_Label'], y=company_raw['Subscribers_Millions'],
                          mode='lines+markers', name='Subscribers', line=dict(color='#3498DB', width=3)),
                row=1, col=1
            )
            
            if len(company_reg) > 0:
                fig.add_trace(
                    go.Bar(x=company_reg['Quarter_Label'],
                          y=company_reg['Subscribers_Millions_yoy_growth_rate'] * 100,
                          name='Actual YoY Growth %',
                          marker_color=np.where(
                              company_reg['Subscribers_Millions_yoy_growth_rate'] >= 0, '#27AE60', '#E74C3C'
                          )),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=company_reg['Quarter_Label'],
                              y=company_reg['Predicted_YoY_Growth'] * 100,
                              mode='lines+markers', name='Predicted YoY Growth %',
                              line=dict(color='#F39C12', width=2, dash='dash')),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Risk probability over time
        if len(company_cls) > 0 and 'Decline_Probability' in company_cls.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=company_cls['Quarter_Label'],
                y=company_cls['Decline_Probability'] * 100,
                mode='lines+markers+text',
                text=[f"{v:.0f}%" for v in company_cls['Decline_Probability'] * 100],
                textposition='top center',
                line=dict(color='#E74C3C', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)',
                name='Deceleration Risk'
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="#F39C12",
                         annotation_text="50% Threshold", annotation_position="top left")
            fig.update_layout(
                height=400,
                yaxis_title="Deceleration Probability (%)",
                yaxis_range=[0, 100],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No classification data available for this company.")
    
    with tab3:
        # Feature contribution analysis
        st.subheader("What's Driving the Prediction?")
        
        if len(company_cls) > 0:
            latest_row = company_cls.iloc[-1]
            cls_features = metadata['classification']['features']
            
            # Get feature values and their contribution (coefficient * scaled value)
            feature_vals = latest_row[cls_features].values.reshape(1, -1)
            scaled_vals = cls_scaler.transform(feature_vals)[0]
            contributions = cls_model.coef_[0] * scaled_vals
            
            driver_df = pd.DataFrame({
                'Feature': [f.replace('_yoy_diff', ' (YoY Δ)').replace('_', ' ').title() for f in cls_features],
                'Raw Value': feature_vals[0],
                'Contribution': contributions
            }).sort_values('Contribution', ascending=True)
            
            fig = px.bar(
                driver_df,
                x='Contribution',
                y='Feature',
                orientation='h',
                color='Contribution',
                color_continuous_scale=['#27AE60', '#ECF0F1', '#E74C3C'],
                color_continuous_midpoint=0
            )
            fig.update_layout(
                height=350,
                xaxis_title="Contribution to Risk Score",
                yaxis_title="",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Positive contributions increase deceleration risk; negative contributions decrease it.")

# ============================================================
# PAGE 3: WHAT-IF SIMULATOR
# ============================================================
elif page == "🧪 What-If Simulator":
    st.markdown('<p class="main-header">What-If Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Adjust external signals and see how the predicted risk changes</p>', unsafe_allow_html=True)
    
    selected_company = st.selectbox("Select a Company", companies, index=companies.index('Netflix'))
    
    company_cls = cls_data[cls_data['Company'] == selected_company].sort_values(['Year', 'Quarter'])
    
    if len(company_cls) > 0:
        latest_row = company_cls.iloc[-1]
        cls_features = metadata['classification']['features']
        
        st.markdown("### Adjust External Signals")
        st.caption("Move the sliders to simulate changes in external factors and see how the risk score responds.")
        
        feature_labels = {
            'consumer_sentiment_yoy_diff': ('Consumer Sentiment (YoY Change)', -30.0, 30.0),
            'cpi_yoy': ('CPI Inflation Rate (YoY %)', -2.0, 12.0),
            'gt_pos_free_trial_yoy_diff': ('Free Trial Search Volume (YoY Change)', -50.0, 50.0),
            'gt_pos_premium_yoy_diff': ('Premium Plan Search Volume (YoY Change)', -50.0, 50.0),
            'gt_pos_sign_up_yoy_diff': ('Sign-up Search Volume (YoY Change)', -50.0, 50.0),
        }
        
        adjusted_values = {}
        cols = st.columns(2)
        for i, feat in enumerate(cls_features):
            label, min_val, max_val = feature_labels.get(feat, (feat, -50.0, 50.0))
            current_val = float(latest_row[feat])
            with cols[i % 2]:
                adjusted_values[feat] = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=current_val,
                    step=0.5,
                    help=f"Current value: {current_val:.2f}"
                )
        
        # Predict with adjusted values
        input_arr = np.array([[adjusted_values[f] for f in cls_features]])
        scaled_input = cls_scaler.transform(input_arr)
        new_prob = cls_model.predict_proba(scaled_input)[0][1] * 100
        
        original_prob = latest_row['Decline_Probability'] * 100 if 'Decline_Probability' in latest_row.index else 50
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Risk", f"{original_prob:.1f}%")
        with col2:
            delta = new_prob - original_prob
            st.metric("Adjusted Risk", f"{new_prob:.1f}%", delta=f"{delta:+.1f}%", delta_color="inverse")
        with col3:
            if new_prob > 60:
                st.error("⚠️ HIGH RISK: Growth deceleration likely")
            elif new_prob > 40:
                st.warning("🟡 MODERATE RISK: Monitor closely")
            else:
                st.success("✅ LOW RISK: Growth trajectory stable")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_prob,
            delta={'reference': original_prob, 'increasing': {'color': '#E74C3C'}, 'decreasing': {'color': '#27AE60'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#2C3E50'},
                'steps': [
                    {'range': [0, 40], 'color': '#E8F8F5'},
                    {'range': [40, 60], 'color': '#FEF9E7'},
                    {'range': [60, 100], 'color': '#FDEDEC'}
                ],
                'threshold': {
                    'line': {'color': '#E74C3C', 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            },
            title={'text': "Deceleration Risk Score"}
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for this company.")

# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transparent evaluation of our prediction models</p>', unsafe_allow_html=True)
    
    tab_reg, tab_cls = st.tabs(["📊 Regression Model", "🎯 Classification Model"])
    
    with tab_reg:
        st.subheader("Predicting YoY Subscriber Growth Rate")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metadata['regression']['metrics']['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metadata['regression']['metrics']['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metadata['regression']['metrics']['mae']:.4f}")
        
        st.markdown("**Features Used:**")
        for f in metadata['regression']['features']:
            st.markdown(f"- `{f}`")
        
        # Actual vs Predicted scatter
        reg_preds_data = pd.read_csv('data/regression_predictions.csv')
        fig = px.scatter(
            reg_preds_data,
            x='Subscribers_Millions_yoy_growth_rate',
            y='Predicted',
            color='Company',
            hover_data=['Company', 'Quarter_Label'],
            labels={
                'Subscribers_Millions_yoy_growth_rate': 'Actual YoY Growth Rate',
                'Predicted': 'Predicted YoY Growth Rate'
            },
            title="Actual vs Predicted (2025 Holdout)"
        )
        # Add perfect prediction line
        min_val = min(reg_preds_data['Subscribers_Millions_yoy_growth_rate'].min(), reg_preds_data['Predicted'].min())
        max_val = max(reg_preds_data['Subscribers_Millions_yoy_growth_rate'].max(), reg_preds_data['Predicted'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color='#E74C3C', dash='dash')
        ))
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement comparison
        st.subheader("📈 Improvement Over Previous Model")
        comp_data = pd.DataFrame({
            'Model': ['Old (QoQ Target)', 'New (YoY Target)'],
            'R²': [-0.15, 0.56],
            'RMSE': [0.117, 0.078]
        })
        fig = px.bar(comp_data, x='Model', y='R²', color='Model',
                    color_discrete_map={'Old (QoQ Target)': '#E74C3C', 'New (YoY Target)': '#27AE60'},
                    text='R²', title="R² Score Comparison")
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=350, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_cls:
        st.subheader("Predicting YoY Growth Rate Decline")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metadata['classification']['metrics']['accuracy']:.1%}")
        with col2:
            st.metric("F1 Score", f"{metadata['classification']['metrics']['f1']:.1%}")
        with col3:
            st.metric("AUC", f"{metadata['classification']['metrics']['auc']:.4f}")
        
        st.markdown("**Features Used:**")
        for f in metadata['classification']['features']:
            st.markdown(f"- `{f}`")
        
        # Confusion matrix
        cls_preds_data = pd.read_csv('data/classification_predictions.csv')
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(cls_preds_data['YoY_Growth_Rate_Declining'], cls_preds_data['Predicted'])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Declining', 'Declining'],
            y=['Not Declining', 'Declining'],
            text_auto=True,
            color_continuous_scale=['#ECF0F1', '#2C3E50'],
            title="Confusion Matrix (2025 Holdout)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement comparison
        st.subheader("📈 Improvement Over Previous Model")
        comp_cls = pd.DataFrame({
            'Metric': ['Accuracy', 'Accuracy', 'F1 Score', 'F1 Score'],
            'Model': ['Old (QoQ)', 'New (YoY)', 'Old (QoQ)', 'New (YoY)'],
            'Score': [65.6, 75.0, 73.1, 79.1]
        })
        fig = px.bar(comp_cls, x='Metric', y='Score', color='Model', barmode='group',
                    color_discrete_map={'Old (QoQ)': '#95A5A6', 'New (YoY)': '#27AE60'},
                    text='Score', title="Classification Performance Comparison")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, yaxis_range=[0, 100], plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7F8C8D; font-size: 0.9rem;">
    <strong>Subscription Fatigue Predictor</strong> | MIDS W210 Capstone | 
    Eric Wu, Mala Ramakrishnan, Mina Mafi, Samuel Dominguez | March 2026
</div>
""", unsafe_allow_html=True)
