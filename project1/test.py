import pandas as pd
import numpy as np 
from sklearn.model_selection  import StratifiedKFold
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics  import f1_score 
from sklearn.compose  import ColumnTransformer 
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing  import StandardScaler 
import lightgbm as lgb 
import catboost as cb 
from sklearn.base  import clone 
import joblib
import optuna
from optuna.samplers  import TPESampler
from category_encoders import TargetEncoder 
from pathlib import Path 
from functools import partial 
 
# ==================== 自定义评估函数 ====================
def macro_f1_eval(y_true, y_pred):
    """ 适配LightGBM的feval格式 """
    y_pred = y_pred.reshape(len(np.unique(y_true)),  -1).argmax(axis=0)
    return 'macro_f1', f1_score(y_true, y_pred, average='macro'), True
 
# ==================== 安全数据加载 ====================
def load_and_preprocess():
    current_dir = Path(__file__).parent 
    train_path = current_dir / "data" / "train_data.csv" 
    test_path = current_dir / "data" / "test_data.csv" 
 
    # 诊断输出示例 
    print(f"[DEBUG] 正在加载数据：\n训练路径：{train_path}\n测试路径：{test_path}")
 
    # 数据加载与校验 
    try:
        train_df = pd.read_csv(train_path).drop(columns=['id'],  errors='ignore')
        X = train_df.drop(columns=['attack_cat']) 
        y = LabelEncoder().fit_transform(train_df['attack_cat'])
        joblib.dump(LabelEncoder(),  'label_encoder.pkl') 
        return X, y 
    except Exception as e:
        raise RuntimeError(f"数据加载失败：{str(e)}") from e 
 
# ==================== 特征工程管线 ====================
def build_feature_pipeline():
    """ 带数据泄露防护的目标编码 """
    return ColumnTransformer([
        ('scaler', StandardScaler(), [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 
            'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
            'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 
            'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean'
        ]),
        ('target_enc', TargetEncoder(), ['proto', 'service', 'state'])
    ], remainder='passthrough')
 
# ==================== 贝叶斯优化框架 ====================
def create_objective(X, y):
    """ 封装优化目标避免数据泄漏 """
    def objective(trial):
        params = {
            'boosting_type': trial.suggest_categorical('boosting_type',  ['gbdt', 'dart']),
            'learning_rate': trial.suggest_float('lr',  0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves',  31, 512, step=16),
            'max_depth': trial.suggest_int('max_depth',  3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples',  20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha',  1e-9, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda',  1e-9, 10.0, log=True),
            'random_state': 42 
        }
 
        model = Pipeline([
            ('preprocessor', build_feature_pipeline()),
            ('classifier', lgb.LGBMClassifier(
                **params,
                objective='multiclass',
                num_class=10,
                metric='None',  # 禁用默认指标 
                feval=macro_f1_eval 
            ))
        ])
 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X,  y):
            try:
                model.fit( 
                    X.iloc[train_idx],  y[train_idx],
                    classifier__eval_set=[(X.iloc[val_idx],  y[val_idx])],
                    classifier__callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                preds = model.predict(X.iloc[val_idx]) 
                scores.append(f1_score(y[val_idx],  preds, average='macro'))
            except Exception as e:
                print(f"交叉验证异常：{str(e)}")
                scores.append(0.0) 
                
        return np.mean(scores) 
    return objective 
 
# ==================== 高性能集成模型 ====================
class EnsembleSystem:
    def __init__(self, weights=None):
        self.models  = []
        self.weights  = weights or [0.6, 0.4]
        self.le  = joblib.load('label_encoder.pkl') 
 
    def add_model(self, model):
        self.models.append(model) 
 
    def train_models(self, X, y):
        """ 全量训练确保测试预测质量 """
        print("[INFO] 开始全量模型训练...")
        for model in self.models: 
            model.fit(X,  y)
 
    def predict_ensemble(self, X_test):
        """ 加权概率集成 """
        proba_list = [model.predict_proba(X_test) for model in self.models] 
        weighted_probas = np.sum([w  * p for w, p in zip(self.weights,  proba_list)], axis=0)
        return self.le.inverse_transform(weighted_probas.argmax(axis=1)) 
 
# ==================== 主流程优化 ====================
def main():
    # 数据准备 
    X, y = load_and_preprocess()
    test_df = pd.read_csv(Path(__file__).parent  / "data" / "test_data.csv") 
    test_ids = test_df.pop('id') 
 
    # 超参数优化
    study = optuna.create_study( 
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner() 
    )
    study.optimize(create_objective(X,  y), n_trials=50, timeout=3600)
 
    # 集成模型构建 
    system = EnsembleSystem()
    
    # LightGBM模型配置过滤后的最优参数
    lgb_params = {k: v for k, v in study.best_params.items()  
                 if k not in ['num_class', 'metric']}
    lgb_params.update({ 
        'objective': 'multiclass',
        'num_class': 10,
        'random_state': 42,
        'deterministic': True 
    })
    
    system.add_model(Pipeline([ 
        ('preprocessor', build_feature_pipeline()),
        ('classifier', lgb.LGBMClassifier(**lgb_params))
    ]))
 
    # CatBoost模型配置（修正参数）
    system.add_model(Pipeline([ 
        ('preprocessor', build_feature_pipeline()),
        ('classifier', cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            random_seed=42,
            silent=True
        ))
    ]))
 
    # 全量训练与预测 
    system.train_models(X,  y)
    final_preds = system.predict_ensemble(test_df) 
 
    # 生成提交文件 
    pd.DataFrame({'id': test_ids, 'attack_cat': final_preds}) \
      .to_csv('submission.csv',  index=False)
    print("[SUCCESS] 提交文件已生成！")
 
if __name__ == "__main__":
    main()