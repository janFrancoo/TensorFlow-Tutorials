import tensorflow.contrib.keras as keras
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = load_wine()
feat_data = data['data']
labels = data['target']

x_train, x_test, y_train, y_test = train_test_split(feat_data, labels, test_size=.3, random_state=101)

scl = MinMaxScaler()
scaled_x_train = scl.fit_transform(x_train)
scaled_x_test = scl.fit_transform(x_test)

model = keras.models.Sequential()
model.add(keras.layers.Dense(units=13, input_dim=13, activation='relu'))
model.add(keras.layers.Dense(units=13, activation='relu'))
model.add(keras.layers.Dense(units=13, activation='relu'))
model.add(keras.layers.Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scaled_x_train, y_train, epochs=50)

predictions = model.predict_classes(scaled_x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
