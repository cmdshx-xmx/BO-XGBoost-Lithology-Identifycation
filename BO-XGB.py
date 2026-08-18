import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import xgboost as xgb 
from imblearn.over_sampling import SMOTE 
from bayes_opt import BayesianOptimization 
import warnings 
import joblib 
import sys 

warnings.filterwarnings('ignore') 

# ===================================================================== 
#FILE_PATH = ' ' 
# ===================================================================== 

print("正在读取数据集...") 
data = pd.read_excel(FILE_PATH) 

target_col = '岩性' 
if target_col not in data.columns: sys.exit("错误：找不到 '岩性' 列！") 

data[target_col] = data[target_col].astype(str).str.strip() \
    .str.replace(r'\(|\（', '(', regex=True) \
    .str.replace(r'\)|\）', ')', regex=True) \
    .str.replace(' ', '')  

# well logs
base_logs = ['GR', 'CNL', 'DEN', 'RT'] 
optional_logs = ['AC', 'T2lm', 'PE'] 

log_features = [f for f in base_logs if f in data.columns] 
for f in optional_logs: 
    if f in data.columns and not data[f].isnull().all(): 
        log_features.append(f) 

final_raw_features = [f for f in log_features if not any(
    kw in f.upper() for kw in ['DEPTH', '深', 'WELL', '井']
)] 

X = data[final_raw_features].copy() 
y = data[target_col] 

imputer = SimpleImputer(strategy='median') 
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=final_raw_features) 


print("\n构建特征集") 

if 'RT' in X_imputed.columns: 
    X_imputed['Log_RT'] = np.log10(X_imputed['RT'] + 0.01) 

if 'AC' in X_imputed.columns and 'DEN' in X_imputed.columns: 
    X_imputed['AC_DEN_Ratio'] = X_imputed['AC'] / (X_imputed['DEN'] + 0.01) 

if 'PE' in X_imputed.columns and 'DEN' in X_imputed.columns: 
    X_imputed['PE_DEN'] = X_imputed['PE'] * X_imputed['DEN'] 

final_feature_cols = X_imputed.columns.tolist() 
print(f"最终冲刺特征维度: {len(final_feature_cols)}") 

le = LabelEncoder() 
y_encoded = le.fit_transform(y) 
num_classes = len(le.classes_) 


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_imputed.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
) 

# normalization
scaler = StandardScaler() 
X_train_full_scaled = scaler.fit_transform(X_train_full) 
X_test_scaled = scaler.transform(X_test) 

# SMOTE
X_xgb, y_xgb = X_train_full_scaled, y_train_full 

min_samples = np.bincount(y_xgb).min() 
safe_k = max(1, min(5, min_samples - 1)) 

smote = SMOTE(random_state=42, k_neighbors=safe_k) 
print(f"应用 SMOTE (k={safe_k})...") 

X_xgb_sm, y_xgb_sm = smote.fit_resample(X_xgb, y_xgb) 
print(f"XGB训练集: {len(y_xgb_sm)}") 

# BO-XGBoost
print("\n" + "-"*60) 
print("开始贝叶斯优化 XGBoost 模型...") 
print("-"*60) 

def xgb_cv(max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma): 
    params = { 
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma,
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False 
    } 

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
    cv_scores = [] 

    for train_idx, val_idx in skf.split(X_xgb_sm, y_xgb_sm): 
        model = xgb.XGBClassifier(**params) 
        model.fit(X_xgb_sm[train_idx], y_xgb_sm[train_idx], verbose=False) 
        y_val_pred = model.predict(X_xgb_sm[val_idx]) 
        cv_scores.append(accuracy_score(y_xgb_sm[val_idx], y_val_pred)) 

    return np.mean(cv_scores) 

optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds={ 
        'max_depth': (6, 12),
        'learning_rate': (0.01, 0.15),
        'n_estimators': (200, 800),
        'min_child_weight': (1, 5),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.5, 0.95),
        'gamma': (0, 0.3) 
    },
    random_state=42
) 

optimizer.maximize(init_points=3, n_iter=12) 
best_params = optimizer.max['params'] 


print("\n训练 XGBoost 模型...") 

final_model = xgb.XGBClassifier( 
    max_depth=int(best_params['max_depth']),
    learning_rate=best_params['learning_rate'],
    n_estimators=int(best_params['n_estimators']),
    min_child_weight=int(best_params['min_child_weight']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
) 

final_model.fit(X_xgb_sm, y_xgb_sm, verbose=False) 
y_pred = final_model.predict(X_test_scaled) 


acc = accuracy_score(y_test, y_pred) 

print(f"\n{'='*30}") 
print(f"测试集准确率: {acc:.4f}") 
print(f"{'='*30}\n") 

print("详细分类报告") 
print(classification_report(y_test, y_pred, target_names=le.classes_)) 

print("="*15 + " 过渡相(含砂/含泥)专项诊断 " + "="*15) 

report = classification_report(
    y_test, y_pred, target_names=le.classes_, output_dict=True
) 

focus = ['含砂', '含泥', '含灰'] 

for cn in le.classes_: 
    if any(f in cn for f in focus): 
        print(
            f"'{cn}': 精确率={report[cn]['precision']:.2f}, "
            f"召回率={report[cn]['recall']:.2f}, "
            f"F1分数={report[cn]['f1-score']:.2f}"
        ) 


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.dpi'] = 150 

lithology_labels = le.classes_.tolist() 
num_classes = len(lithology_labels) 
cell_size = 0.95 
fig_size = max(11, num_classes * cell_size + 3) 

fig = plt.figure(figsize=(fig_size, fig_size)) 
ax = fig.add_axes([0.22, 0.26, 0.65, 0.64]) 

cm = confusion_matrix(y_test, y_pred) 

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=lithology_labels,
    yticklabels=lithology_labels,
    ax=ax,
    square=True,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={'shrink': 0.65, 'label': '样本数量'},
    annot_kws={'size': 10}
) 

ax.set_title('岩性识别混淆矩阵', fontsize=16, pad=15) 
ax.set_xlabel('预测岩性', fontsize=13, labelpad=15) 
ax.set_ylabel('真实岩性', fontsize=13, labelpad=15) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=10) 
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10) 

plt.savefig('BO-XGBoost_SMOTE混淆矩阵.png', dpi=600, bbox_inches='tight') 
plt.show() 


joblib.dump(final_model, 'model_imbalanced.pkl') 
joblib.dump(scaler, 'scaler_imbalanced.pkl') 
joblib.dump(le, 'encoder_imbalanced.pkl') 

print("\n模型及配件已保存至本地。")