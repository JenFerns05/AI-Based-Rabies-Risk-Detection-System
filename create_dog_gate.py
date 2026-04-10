import tensorflow as tf
import tf2onnx
import numpy as np

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

# Freeze base
base_model.trainable = False

# Add binary head: dog vs not-dog
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(base_model.input, output)

# Dummy input for ONNX export
spec = (tf.TensorSpec((1,224,224,3), tf.float32, name="input"),)

# Convert to ONNX
tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path="dog_gate.onnx",
    opset=13
)

print("✅ dog_gate.onnx created")
