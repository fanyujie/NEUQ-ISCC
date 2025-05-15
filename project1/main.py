
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
from sklearn.base import clone
import joblib
import optuna
from optuna.samplers import TPESampler
from category_encoders import TargetEncoder

# ==================== 自定义宏F1评估函数 ====================
def macro_f1_eval(y_true, y_pred):
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return f1_score(y_true, y_pred, average='macro')

# ==================== 数据预处理 ====================
def load_and_preprocess():
    import os
    from pathlib import Path
    
    # 动态构建路径
    current_dir = Path(__file__).parent  # 获取当前脚本所在目录
    train_path = current_dir / "data" / "train_data.csv"
    test_path = current_dir / "data" / "test_data.csv"
    
    # 路径诊断输出
    print(f"当前脚本位置：{current_dir}")
    print(f"训练文件预期路径：{train_path}")
    print(f"测试文件预期路径：{test_path}")
    
    if not train_path.exists():
        raise FileNotFoundError(f"训练数据未在以下路径找到：{train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"测试数据未在以下路径找到：{test_path}")

    # 加载数据（使用更安全的路径构建方式）
    train_df = pd.read_csv(str(train_path))  # 转换为字符串兼容旧版本pandas
    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)
    
    X = train_df.drop('attack_cat', axis=1)
    y = train_df['attack_cat']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, 'label_encoder.pkl')
    return X, y_encoded, le

# ==================== 贝叶斯优化目标函数 ====================
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'custom',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.2),
        'class_weight': 'balanced'
    }
    
    model = Pipeline([
        ('preprocessor', build_feature_pipeline()),
        ('classifier', lgb.LGBMClassifier(**params))
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        scores.append(f1_score(y_val, val_pred, average='macro'))
    
    return np.mean(scores)

# ==================== 优化后的特征工程 ====================
def build_feature_pipeline():
    categorical_features = ['proto', 'service', 'state']
    numerical_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
                         'rate', 'sttl', 'dttl', 'sload', 'dload',
                         'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
                         'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
                         'ackdat', 'smean', 'dmean']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', TargetEncoder(), categorical_features)  # 目标编码替代One-Hot
        ],
        remainder='passthrough'
    )
    return preprocessor

# ==================== 集成模型构建函数 ====================
def build_ensemble_model(best_params):  # 添加参数传递
    # LightGBM模型使用贝叶斯优化的参数
    lgb_model = Pipeline([
        ('preprocessor', build_feature_pipeline()),
        ('classifier', lgb.LGBMClassifier(**best_params))  # 正确引用参数
    ])
    
    # CatBoost模型使用默认参数
    cb_model = Pipeline([
        ('preprocessor', build_feature_pipeline()),
        ('classifier', cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            verbose=False,
            class_weights='Balanced'
        ))
    ])
    
    return [lgb_model, cb_model]

# ==================== 主流程 ====================
def main():
    X, y, le = load_and_preprocess()
    
    # 定义唯一的objective函数（修复重复定义问题）
    def objective(trial):
        params = {
            'objective': 'multiclass',
            'num_class': 10,
            'metric': 'custom',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.2),
            'class_weight': 'balanced'
        }
        
        model = Pipeline([
            ('preprocessor', build_feature_pipeline()),
            ('classifier', lgb.LGBMClassifier(**params))
        ])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []  # 显式初始化
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                scores.append(f1_score(y_val, val_pred, average='macro'))
            except Exception as e:
                print(f"交叉验证失败: {e}")
                scores.append(0.0)  # 避免空列表
        
        return np.mean(scores) if len(scores) > 0 else 0.0
    
    # 超参数优化
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    
    # 关键修正：传递best_params到集成模型
    models = build_ensemble_model(best_params)
    final_preds = []
    
    for model in models:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 初始化全样本概率矩阵
        oof_probs = np.zeros((len(X), 10))  # 形状为(总样本数, 类别数)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            
            # 关键修正：直接填充对应验证集位置的预测概率
            val_probs = cloned_model.predict_proba(X_val)
            oof_probs[val_idx] = val_probs  # 按索引填充
        
        final_preds.append(oof_probs)  # 存储全体样本概率
    
    # 加权集成预测结果
    ensemble_probs = 0.6 * final_preds[0] + 0.4 * final_preds[1]
    final_labels = np.argmax(ensemble_probs, axis=1)
    
    # 生成提交文件
    test_df = pd.read_csv('data\\test_data.csv')
    test_ids = test_df['id']
    
    # 对测试集进行预测
    test_preds = []
    for model in models:
        test_preds.append(model.predict_proba(test_df.drop('id', axis=1)))
    ensemble_test_probs = 0.6 * test_preds[0] + 0.4 * test_preds[1]
    test_labels = le.inverse_transform(np.argmax(ensemble_test_probs, axis=1))
    
    pd.DataFrame({'id': test_ids, 'attack_cat': test_labels})\
      .to_csv('submission.csv', index=False, encoding='utf-8')
    print("提交文件生成完成")

if __name__ == "__main__":
    main()