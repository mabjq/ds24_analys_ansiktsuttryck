from tensorflow import keras

# Ladda modellen
model = keras.models.load_model('/home/mabjq/ml_projects/kunskapskontroll_DL/modelv1.keras')

# Visa modellens sammanfattning
model.summary()

# Visa input- och output-form
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")