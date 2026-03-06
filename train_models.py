"""
Train and save the best regression and classification models for the Streamlit dashboard.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score, classification_report)
import joblib
import json

# Load data
df = pd.read_csv('data/dataset_16_companies_with_yoy_diff.csv')

# ============================================================
# REGRESSION MODEL: Predict YoY Growth Rate
# Best 7 features: R² = 0.56
# ============================================================
reg_features = [
    'cpi_all_urban_consumers',
    'gt_neg_alternative_yoy_diff',
    'gt_neg_not_working',
    'gt_pos_free_trial_yoy_diff',
    'real_disposable_personal_income_yoy_diff',
    'reddit_post_count',
    'gt_pos_premium'
]
reg_target = 'Subscribers_Millions_yoy_growth_rate'

reg_df = df.dropna(subset=[reg_target] + reg_features)
train_reg = reg_df[reg_df['Year'] < 2025]
test_reg = reg_df[reg_df['Year'] == 2025]

X_train_reg = train_reg[reg_features]
y_train_reg = train_reg[reg_target]
X_test_reg = test_reg[reg_features]
y_test_reg = test_reg[reg_target]

reg_scaler = StandardScaler()
X_train_reg_scaled = reg_scaler.fit_transform(X_train_reg)
X_test_reg_scaled = reg_scaler.transform(X_test_reg)

reg_model = LinearRegression()
reg_model.fit(X_train_reg_scaled, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg_scaled)
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"Regression: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

joblib.dump(reg_model, 'models/regression_model.joblib')
joblib.dump(reg_scaler, 'models/regression_scaler.joblib')

# ============================================================
# CLASSIFICATION MODEL: Predict YoY Growth Rate Declining
# Best accuracy model: C=10.0, class_weight='balanced'
# Features: consumer_sentiment_yoy_diff, cpi_yoy, gt_pos_free_trial_yoy_diff,
#           gt_pos_premium_yoy_diff, gt_pos_sign_up_yoy_diff
# Acc=0.75, F1=0.77, AUC=0.77
# ============================================================
df_sorted = df.sort_values(['Company', 'Year', 'Quarter']).reset_index(drop=True)
df_sorted['prev_yoy_growth_rate'] = df_sorted.groupby('Company')['Subscribers_Millions_yoy_growth_rate'].shift(1)
df_sorted['YoY_Growth_Rate_Declining'] = (
    df_sorted['Subscribers_Millions_yoy_growth_rate'] < df_sorted['prev_yoy_growth_rate']
).astype(float)
df_sorted.loc[df_sorted['prev_yoy_growth_rate'].isna(), 'YoY_Growth_Rate_Declining'] = np.nan

cls_features = [
    'consumer_sentiment_yoy_diff',
    'cpi_yoy',
    'gt_pos_free_trial_yoy_diff',
    'gt_pos_premium_yoy_diff',
    'gt_pos_sign_up_yoy_diff'
]
cls_target = 'YoY_Growth_Rate_Declining'

cls_df = df_sorted.dropna(subset=[cls_target] + cls_features)
train_cls = cls_df[cls_df['Year'] < 2025]
test_cls = cls_df[cls_df['Year'] == 2025]

X_train_cls = train_cls[cls_features]
y_train_cls = train_cls[cls_target]
X_test_cls = test_cls[cls_features]
y_test_cls = test_cls[cls_target]

cls_scaler = StandardScaler()
X_train_cls_scaled = cls_scaler.fit_transform(X_train_cls)
X_test_cls_scaled = cls_scaler.transform(X_test_cls)

# KEY: Use C=10.0 and class_weight='balanced' as found in the experiment
cls_model = LogisticRegression(C=10.0, class_weight='balanced', max_iter=1000, random_state=42)
cls_model.fit(X_train_cls_scaled, y_train_cls)

y_pred_cls = cls_model.predict(X_test_cls_scaled)
y_prob_cls = cls_model.predict_proba(X_test_cls_scaled)[:, 1]
acc = accuracy_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)
auc = roc_auc_score(y_test_cls, y_prob_cls)
print(f"Classification: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
print(classification_report(y_test_cls, y_pred_cls))

joblib.dump(cls_model, 'models/classification_model.joblib')
joblib.dump(cls_scaler, 'models/classification_scaler.joblib')

# ============================================================
# SAVE MODEL METADATA
# ============================================================
metadata = {
    'regression': {
        'features': reg_features,
        'target': reg_target,
        'metrics': {'r2': round(r2, 4), 'rmse': round(rmse, 4), 'mae': round(mae, 4)},
        'description': 'OLS Linear Regression predicting YoY subscriber growth rate'
    },
    'classification': {
        'features': cls_features,
        'target': cls_target,
        'metrics': {'accuracy': round(acc, 4), 'f1': round(f1, 4), 'auc': round(auc, 4)},
        'hyperparameters': {'C': 10.0, 'class_weight': 'balanced'},
        'description': 'Logistic Regression predicting whether YoY growth rate is declining'
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# SAVE PREDICTIONS FOR DASHBOARD
# ============================================================
# Regression predictions
reg_preds = test_reg[['Company', 'Year', 'Quarter', 'Quarter_Label', reg_target]].copy()
reg_preds['Predicted'] = y_pred_reg
reg_preds.to_csv('data/regression_predictions.csv', index=False)

# Classification predictions
cls_preds = test_cls[['Company', 'Year', 'Quarter', 'Quarter_Label', cls_target]].copy()
cls_preds['Predicted'] = y_pred_cls
cls_preds['Probability'] = y_prob_cls
cls_preds.to_csv('data/classification_predictions.csv', index=False)

# Full dataset with predictions for all rows
full_reg = df.dropna(subset=[reg_target] + reg_features).copy()
X_full = reg_scaler.transform(full_reg[reg_features])
full_reg['Predicted_YoY_Growth'] = reg_model.predict(X_full)
full_reg.to_csv('data/full_regression_data.csv', index=False)

full_cls = df_sorted.dropna(subset=[cls_target] + cls_features).copy()
X_full_cls = cls_scaler.transform(full_cls[cls_features])
full_cls['Predicted_Declining'] = cls_model.predict(X_full_cls)
full_cls['Decline_Probability'] = cls_model.predict_proba(X_full_cls)[:, 1]
full_cls.to_csv('data/full_classification_data.csv', index=False)

print(f"\nAll models and data saved successfully!")
print(f"Regression predictions: {len(reg_preds)} rows")
print(f"Classification predictions: {len(cls_preds)} rows")
print(f"Full regression data: {len(full_reg)} rows")
print(f"Full classification data: {len(full_cls)} rows")
