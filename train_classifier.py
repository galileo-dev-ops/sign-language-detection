import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


with open('model.p', 'rb') as f:
    model_obj = pickle.load(f)

# Recreate the TensorFlow model from the loaded object
model_tf = tf.keras.models.model_from_json(model_obj.to_json())
model_tf.set_weights(model_obj.get_weights())

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
