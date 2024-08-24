import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import os

tf.get_logger().setLevel('ERROR')

# 設置圖像大小和批次大小
img_height, img_width = 224, 224
batch_size = 32

# 訓練數據增強
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 加載訓練數據，並自動調整圖片大小
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# 驗證數據
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# 加載 MobileNetV2 模型並去除頂部的全連接層
base_model = MobileNetV2(weights="mobilenet_v2_weights.h5", include_top=False, input_shape=(img_height, img_width, 3))

# 冻结 base_model 的所有层
base_model.trainable = False

# 添加自定义的分类层
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 2类分类问题
])

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
epochs = 10  # 訓練輪數，你可以根據需要調整

# 設置保存模型的資料夾路徑
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

best_val_accuracy = 0.0
best_model_path = ''

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=1,  # 每次只訓練一個 epoch
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # 保存當前輪次的模型
    model_path = os.path.join(model_dir, f'epoch_{epoch + 1}.h5')
    model.save(model_path, include_optimizer=False)
    print(f"Model saved at {model_path}")

    # 檢查當前模型的驗證集表現，如果是最好的，就保存 TFLite 模型
    val_accuracy = history.history['val_accuracy'][-1]
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_path = model_path

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        with open('cat_dog_classifier_best.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Best models converted to TFLite and saved.")

# 打印最佳模型信息
print(f"Best models found at epoch: {best_model_path}, with validation accuracy: {best_val_accuracy:.4f}")

# 打印模型架構
model.summary()

# 最終保存的最佳模型 (Keras 格式)
model.save('classifier.h5', include_optimizer=False)
