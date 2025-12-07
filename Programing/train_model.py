import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------

df = pd.read_csv("asl_mediapipe_keypoints_dataset.csv")

print("Dataset Loaded ✓")
print(df.head())

# آخر عمود هو الـ Label (الحرف)
labels = df.iloc[:, -1].values
X = df.iloc[:, :-1].values

# ---------------------------------------------------
# 2) Encode labels (A,B,C,...)
# ---------------------------------------------------

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)

num_classes = y.shape[1]
print("Classes Count:", num_classes)


# ---------------------------------------------------
# 3) Train / Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test Split Done ✓")


# ---------------------------------------------------
# 4) Build Model
# ---------------------------------------------------

model = Sequential([
    Dense(256, activation='relu', input_shape=( X.shape[1],)),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------
# 5) Train Model
# ---------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32
)

# ---------------------------------------------------
# 6) Save Model
# ---------------------------------------------------

model.save("asl_keypoints_model.h5")
np.save("label_classes.npy", label_encoder.classes_)

print("\n====================================")
print("🎉 Training Finished! Model Saved ✓")
print("====================================")
