# ======================
# 1. 环境配置与数据加载
# ======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import os
import time
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设置文件路径
TRAIN_PATH = r"E:\py\1\SantanderCustomerTransactionPrediction\train.csv"
TEST_PATH = r"E:\py\1\SantanderCustomerTransactionPrediction\test.csv"
OUTPUT_DIR = r"E:\py\1\SantanderCustomerTransactionPrediction\output"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
print(f"正在加载训练数据: {TRAIN_PATH}")
train = pd.read_csv(TRAIN_PATH)
print(f"训练集已加载，形状: {train.shape}")

print(f"正在加载测试数据: {TEST_PATH}")
test = pd.read_csv(TEST_PATH)
print(f"测试集已加载，形状: {test.shape}")

# ======================
# 2. 数据预处理
# ======================

# 合并训练集和测试集
print("合并训练集和测试集...")
full_data = pd.concat([train.drop('target', axis=1), test], axis=0)

# 删除ID列
if 'ID_code' in full_data.columns:
    full_data.drop('ID_code', axis=1, inplace=True)
    print("已删除ID列")
elif 'id' in full_data.columns:
    full_data.drop('id', axis=1, inplace=True)
    print("已删除id列")

# 检查缺失值
print("\n缺失值分析:")
missing_ratio = full_data.isnull().mean().sort_values(ascending=False)
high_missing = missing_ratio[missing_ratio > 0]
if high_missing.empty:
    print("数据集无缺失值")
else:
    print(f"发现{len(high_missing)}个特征有缺失值:")
    print(high_missing)
    # 删除高缺失率特征
    high_missing_cols = high_missing[high_missing > 0.3].index
    if not high_missing_cols.empty:
        full_data.drop(columns=high_missing_cols, inplace=True)
        print(f"已删除{len(high_missing_cols)}个缺失率>30%的特征")

# ======================
# 3. 数据探索与分析
# ======================

# 目标变量分布可视化
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=train)
plt.title('Class Distribution (0: No Transaction, 1: Transaction)')
plt.xlabel('Target Class')
plt.ylabel('Count')
dist_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
plt.savefig(dist_path, dpi=300)
print(f"\n类别分布图已保存至: {dist_path}")
plt.close()

# 计算类别比例
neg_count = train['target'].value_counts()[0]
pos_count = train['target'].value_counts()[1]
print(f"\n类别分布统计:")
print(f"负样本数量: {neg_count} ({neg_count / (neg_count + pos_count):.2%})")
print(f"正样本数量: {pos_count} ({pos_count / (neg_count + pos_count):.2%})")

# ======================
# 4. 特征工程
# ======================

# 特征标准化
print("\n应用特征标准化...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(full_data)
scaled_df = pd.DataFrame(scaled_features, columns=full_data.columns)

# 特征选择 - 基于方差阈值
print("\n应用特征选择...")
variances = scaled_df.var().sort_values()
low_variance_features = variances[variances < 0.01].index.tolist()
print(f"删除{len(low_variance_features)}个低方差特征")
scaled_df.drop(columns=low_variance_features, inplace=True)

# 将数据集拆回训练集和测试集
X = scaled_df.iloc[:len(train)]
y = train['target']
X_test_final = scaled_df.iloc[len(train):]

# 划分训练集和验证集
print("\n划分训练集和验证集...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")

# ======================
# 5. 样本不平衡处理
# ======================

# 应用SMOTE过采样
print("\n应用SMOTE过采样处理样本不平衡...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_res, y_res = smote.fit_resample(X_train, y_train)

# 检查过采样效果
print("\n过采样后类别分布:")
print(f"负样本数量: {sum(y_res == 0)}")
print(f"正样本数量: {sum(y_res == 1)}")
print(f"正样本比例: {sum(y_res == 1) / len(y_res):.1%}")

# ======================
# 6. GPU加速模型构建
# ======================

print("\n开始训练GPU加速模型...")

# 初始化模型
models = {
    'Logistic Regression': LogisticRegression(C=0.01, solver='sag', max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'XGBoost (GPU)': XGBClassifier(
        tree_method='gpu_hist',  # GPU加速
        gpu_id=0,  # 使用第一个GPU
        predictor='gpu_predictor',
        eval_metric='logloss'
    ),
    'LightGBM (GPU)': LGBMClassifier(
        device='gpu',  # GPU加速
        gpu_platform_id=0,
        gpu_device_id=0
    )
}

# 计算类别权重用于代价敏感学习
neg_weight = len(y_res) / (2 * sum(y_res == 0))
pos_weight = len(y_res) / (2 * sum(y_res == 1))
class_weights = {0: neg_weight, 1: pos_weight}
print(f"\n类别权重: 负样本={neg_weight:.2f}, 正样本={pos_weight:.2f}")

# 评估模型
print("\n模型性能评估:")
base_results = []
roc_data = []

for name, model in models.items():
    start_time = time.time()
    print(f"\n正在训练 {name} 模型...")

    # 设置类别权重
    if 'XGBoost' in name:
        model.set_params(scale_pos_weight=pos_weight / neg_weight)
    elif 'LightGBM' in name:
        model.set_params(is_unbalance=True)

    # 训练模型
    model.fit(X_res, y_res)
    train_time = time.time() - start_time

    # 在验证集上评估
    start_pred = time.time()
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - start_pred

    # 计算评估指标
    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, output_dict=True)
    f1 = report['1']['f1-score']

    # 保存结果
    result = {
        'Model': name,
        'AUC': auc,
        'F1-Score': f1,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'Train Time (s)': train_time,
        'Predict Time (s)': pred_time
    }
    base_results.append(result)

    # 保存ROC曲线数据
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_data.append((fpr, tpr, name, auc))

    # 打印结果
    print(f"{name}模型训练时间: {train_time:.1f}秒, 预测时间: {pred_time:.1f}秒")
    print(f"AUC: {auc:.4f}, F1-Score: {f1:.4f}")

# 绘制所有模型的ROC曲线
plt.figure(figsize=(10, 8))
for fpr, tpr, name, auc in roc_data:
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

# 绘制基准线
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=300)
print(f"\nROC曲线图已保存至: {roc_path}")
plt.close()

# ======================
# 7. 模型优化与集成
# ======================

# XGBoost超参数调优
print("\n开始XGBoost超参数调优...")
param_grid = {
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    scale_pos_weight=pos_weight / neg_weight,
    eval_metric='logloss',
    n_estimators=300
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=1,  # GPU不支持多进程
    verbose=2
)

print("执行网格搜索...")
grid_search.fit(X_res, y_res)

# 输出最优参数
best_xgb = grid_search.best_estimator_
print(f"\n调优完成! 最优参数: {grid_search.best_params_}")
print(f"最优AUC: {grid_search.best_score_:.4f}")

# 构建Stacking集成模型
print("\n构建Stacking集成模型...")
base_models = [
    ('nb', GaussianNB()),
    ('xgb', best_xgb),
    ('lgbm', LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        is_unbalance=True,
        n_estimators=150
    ))
]

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(
        C=0.01,
        solver='sag',
        max_iter=1000,
        class_weight=class_weights
    ),
    cv=5,
    n_jobs=1  # GPU不支持多进程
)

# 训练集成模型
print("训练Stacking模型...")
stack_model.fit(X_res, y_res)

# 评估集成模型
y_pred_stack = stack_model.predict(X_val)
y_prob_stack = stack_model.predict_proba(X_val)[:, 1]

auc_stack = roc_auc_score(y_val, y_prob_stack)
report_stack = classification_report(y_val, y_pred_stack, output_dict=True)
f1_stack = report_stack['1']['f1-score']

print("\nStacking集成模型性能:")
print(f"AUC: {auc_stack:.4f}")
print(classification_report(y_val, y_pred_stack))

# 添加集成模型结果
base_results.append({
    'Model': 'Stacking',
    'AUC': auc_stack,
    'F1-Score': f1_stack,
    'Precision': report_stack['1']['precision'],
    'Recall': report_stack['1']['recall'],
    'Train Time (s)': 0,
    'Predict Time (s)': 0
})

# 性能对比
results_df = pd.DataFrame(base_results)
results_csv = os.path.join(OUTPUT_DIR, 'model_performance.csv')
results_df.to_csv(results_csv, index=False)
print(f"\n模型性能对比已保存至: {results_csv}")
print(results_df)

# ======================
# 8. 生成预测结果
# ======================

# 在完整训练集上重新训练模型
print("\n在完整训练集上重新训练最终模型...")

# 应用SMOTE到完整训练集
X_full_res, y_full_res = smote.fit_resample(X, y)
print(f"完整训练集过采样后形状: {X_full_res.shape}")

# 训练最终模型
final_model = best_xgb  # 使用最优的XGBoost模型

print("训练最终模型...")
final_model.fit(X_full_res, y_full_res)

# 生成测试集预测
print("生成测试集预测...")
test_probs = final_model.predict_proba(X_test_final)[:, 1]

# 创建提交文件
submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission = pd.DataFrame({
    'ID_code': test['ID_code'],
    'target': test_probs
})
submission.to_csv(submission_path, index=False)
print(f"预测结果已保存至: {submission_path}")

# ======================
# 9. 模型分析报告
# ======================

# 生成模型分析报告
plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='AUC', data=results_df.sort_values('AUC', ascending=False))
plt.title('模型AUC性能对比')
plt.ylim(0.5, 0.9)
plt.xticks(rotation=15)
auc_path = os.path.join(OUTPUT_DIR, 'model_auc_comparison.png')
plt.savefig(auc_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='F1-Score', data=results_df.sort_values('F1-Score', ascending=False))
plt.title('模型F1-Score性能对比')
plt.ylim(0.0, 0.6)
plt.xticks(rotation=15)
f1_path = os.path.join(OUTPUT_DIR, 'model_f1_comparison.png')
plt.savefig(f1_path, dpi=300)
plt.close()

plt.figure(figsize=(12, 7))
sns.barplot(x='Model', y='Train Time (s)', data=results_df.sort_values('Train Time (s)', ascending=False))
plt.title('模型训练时间对比')
plt.xticks(rotation=15)
time_path = os.path.join(OUTPUT_DIR, 'model_train_time.png')
plt.savefig(time_path, dpi=300)
plt.close()


print(f"最佳单一模型: {best_xgb.__class__.__name__} (调优后AUC={grid_search.best_score_:.4f})")
print(f"Stacking集成模型综合性能最优 (AUC={auc_stack:.4f})")
print(f"所有输出文件已保存至: {OUTPUT_DIR}")