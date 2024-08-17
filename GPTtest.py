import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam

tf.get_logger().setLevel('ERROR')

# 設置圖像大小和批次大小
img_height, img_width = 224, 224
batch_size = 32

# 訓練數據增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
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
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# 加載 MobileNetV2 模型並去除頂部的全連接層
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(img_height, img_width, 3))

# 冻结 base_model 的所有层
base_model.trainable = True

# 添加自定义的分类层
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # 2类分类问题
])

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
epochs = 10  # 訓練輪數，你可以根據需要調整

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# 將 Keras 模型轉換為 TFLite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

model.summary()

# 保存 TFLite 模型到文件
with open('cat_dog_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
model.save('classifier.h5', include_optimizer=False)
