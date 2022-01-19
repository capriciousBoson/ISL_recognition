import tensorflow as tf
from utils import load_image

model = tf.keras.models.load_model('trained_model/isl_model_v1.2')
samples_to_predict = load_image()
predictions = model.predict(samples_to_predict)
alphabets = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
             "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

for i in range(35):
    print(" Probability of image belonging to class: ", alphabets[i]," = ", round(predictions[0][i], 3)*100)


