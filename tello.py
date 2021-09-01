import tensorflow as tf
from tensorflow import keras
# Reading the model from JSON file
with open('pydnet.json', 'r') as json_file:
    json_savedModel = json_file.read()
# load the model architecture
model_j = keras.models.model_from_json(json_savedModel)
model_j.summary()
