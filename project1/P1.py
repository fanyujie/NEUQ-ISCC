import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalMaxPool1D,
    Dense, Reshape, MultiHeadAttention, LayerNormalization,
    Dropout, Add, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------------
# 1. 数据预处理（保持不变）
# ------------------------------

# 加载数据
train_data = pd.read_csv("D:/ISCC/attack/train_data.csv")
test_data = pd.read_csv("D:/ISCC/attack/test_data.csv")

# 处理缺失值
for df in [train_data, test_data]:
    df["service"] = df["service"].replace("-", "unknown")

# 分离特征和标签
X_train = train_data.drop(columns=["id", "attack_cat"])
y_train = train_data["attack_cat"]
X_test = test_data.drop(columns=["id"])

# 定义特征类型
categorical_cols = ["proto", "service", "state"]
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# 预处理Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 转换数据
X_train_processed = preprocessor.fit_transform(X_train).toarray()
X_test_processed = preprocessor.transform(X_test).toarray()

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# ------------------------------
# 2. 模型构建（融合Transformer）
# ------------------------------

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Transformer编码器模块"""
    # 自注意力机制
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    res = Add()([inputs, x])  # 残差连接
    x = LayerNormalization(epsilon=1e-6)(res)
    
    # 前馈网络
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([res, x])       # 残差连接
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

n_features = X_train_processed.shape[1]
n_classes = len(label_encoder.classes_)

# 输入层
input_layer = Input(shape=(n_features,))
x = Reshape((n_features, 1))(input_layer)

# 第一阶段：CNN提取局部特征
x = Conv1D(64, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(256, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(256, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(256, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(512, kernel_size=3, activation="relu", padding="valid")(x)
x = MaxPooling1D(pool_size=2)(x)

# 第二阶段：Transformer建模全局依赖
x = transformer_encoder(
    x, 
    head_size=128,   # 注意力头维度
    num_heads=64,    # 注意力头数量
    ff_dim=512,     # 前馈网络维度
    dropout=0.2      # 随机失活概率
)

# 第三阶段：分类头
x = GlobalMaxPool1D()(x)
x = Dense(512, activation="relu")(x)    
x = Dropout(0.5)(x)
output = Dense(n_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ------------------------------
# 3. 模型训练（参数优化）
# ------------------------------

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    "balanced",
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weights = dict(enumerate(class_weights))

# 划分验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train_encoded, 
    test_size=0.2, 
    stratify=y_train_encoded,
    random_state=42
)

# 回调函数
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("attack_model.h5", save_best_only=True)
]

# 训练模型
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=100,
    batch_size=128,  # 减小批次大小以适配Transformer计算
    callbacks=callbacks,
    class_weight=class_weights
)

# ------------------------------
# 4. 测试集预测（保持不变）
# ------------------------------

model.load_weights("attack_model.h5")
y_pred_proba = model.predict(X_test_processed)
y_pred = np.argmax(y_pred_proba, axis=1)
attack_cat_pred = label_encoder.inverse_transform(y_pred)

submission = pd.DataFrame({
    "id": test_data["id"],
    "attack_cat": attack_cat_pred
})
submission["attack_cat"] = submission["attack_cat"].apply(
    lambda x: x if x in label_encoder.classes_ else "Normal"
)
submission.to_csv("submission.csv", index=False, encoding="utf-8")
print("融合模型提交文件已生成：submission.csv")