import tensorflow as tf
import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image

# 載入 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="cat_dog_classifier.tflite")
interpreter.allocate_tensors()

# 獲取輸入和輸出張量的詳細信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 定義圖片大小
img_height, img_width = 224, 224


def predict_image(img):
    # 將圖片調整為模型的輸入大小
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # 將圖片數據傳給模型
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # 獲取模型的輸出結果
    output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # 判斷結果是貓還是狗
    if output_data > 0.5:
        return "Dog"
    else:
        return "Cat"


# 建立 Gradio 界面
interface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type="pil"),
                         outputs="text",
                         title="Cat vs Dog Classifier")

# 啟動 Gradio 應用
interface.launch()
