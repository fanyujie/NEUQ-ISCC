import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
df = pd.read_csv('D:/ISCC/基于网络流量的恶意攻击检测/train_data.csv')

# 删除无关列（如id）
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 定义特征和标签，并保存LabelEncoder实例
le = LabelEncoder()
X = df.drop('attack_cat', axis=1)
y = le.fit_transform(df['attack_cat'])  # 使用保存的LabelEncoder实例

# 定义预处理步骤
categorical_features = ['proto', 'service', 'state']
numerical_features = X.columns.difference(categorical_features).tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建管道并训练模型（增加max_iter至2000）
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,  # 增加迭代次数
        class_weight='balanced'
    ))
])

pipeline.fit(X_train, y_train)

# 评估模型
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# 使用保存的LabelEncoder实例的classes_
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))