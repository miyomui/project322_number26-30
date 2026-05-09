import tensorflow as tf

try:
    model = tf.keras.models.load_model(r'c:\project_aie322\model\thai_digits_cnn_model_best.keras')
    model.summary()
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
except Exception as e:
    print("Error loading model:", e)
