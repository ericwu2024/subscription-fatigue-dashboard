# Subscription Fatigue Predictor

**Early Warning Signals for the Streaming Economy**

An interactive dashboard that uses external signals (Google Trends, Reddit sentiment, macroeconomic indicators) to predict subscriber growth deceleration across 16 major subscription companies — without requiring any internal company data.

## Features

- **Market Overview**: Risk heatmap of all 16 companies color-coded by deceleration probability
- **Company Deep Dive**: Individual company analysis with growth trends, risk timeline, and driver analysis
- **What-If Simulator**: Interactive sliders to simulate how changes in external signals affect risk scores
- **Model Performance**: Transparent evaluation with actual vs predicted plots and improvement comparisons

## Models

| Task | Target | Best Metric |
|------|--------|-------------|
| Regression | YoY Subscriber Growth Rate | R² = 0.56 |
| Classification | Is YoY Growth Rate Declining? | Accuracy = 75%, F1 = 77% |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team

Eric Wu, Mala Ramakrishnan, Mina Mafi, Samuel Dominguez

*UC Berkeley MIDS W210 Capstone — March 2026*
