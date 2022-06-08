import tensorflow as tf
import numpy as np
from LungCancerDetection.settings import BASE_DIR

# with open('test_images.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     images = pickle.load(file)
# with open('test_label.pkl', 'rb') as file2:
#     labels = pickle.load(file2)

def prediction(model_path,image):
    model = tf.keras.models.load_model(model_path)
    img_array = np.reshape(image,(1,64,64,1))
    predict = model.predict(img_array)
    percentage = predict[0][1]*100
    return percentage