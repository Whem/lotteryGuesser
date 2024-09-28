import json

# import numpy as np
# from keras import layers
# from tensorflow import keras

# from algorithms.models import lg_lottery_winner_number
#
#
# # Function to load data from a file and preprocess it
# def load_data(lottery_type):
#     winning_numbers = lg_lottery_winner_number.objects.filter(lottery_type=lottery_type).values_list(
#         'lottery_type_number', flat=True)
#
#     int_array_data = []
#     for item in winning_numbers:
#         int_array_data.append(item)
#
#     # Load data from file, ignoring white spaces and accepting unlimited length numbers
#     data = np.array(int_array_data)
#     # Split data into training and validation sets
#     train_data = data[:int(0.8 * len(data))]
#     val_data = data[int(0.8 * len(data)):]
#     # Get the maximum value in the data
#     max_value = np.max(data)
#     last_data = data[-1]
#     return train_data, val_data, max_value, last_data
#
#
# # Function to create the model
# def create_model(num_features, max_value):
#     # Create a sequential model
#     model = keras.Sequential()
#     # Add an Embedding layer, LSTM layer, and Dense layer to the model
#     model.add(layers.Embedding(input_dim=max_value + 1, output_dim=64))
#     model.add(layers.LSTM(256))
#     model.add(layers.Dense(num_features, activation='softmax'))
#     # Compile the model with categorical crossentropy loss, adam optimizer, and accuracy metric
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# # Function to train the model
# def train_model(model, train_data, val_data):
#     # Fit the model on the training data and validate on the validation data for 100 epochs
#     history = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100)
#
#
# # Function to predict numbers using the trained model
# def predict_numbers(model, val_data, num_features):
#     # Predict on the validation data using the model
#     predictions = model.predict(val_data)
#     # Get the indices of the top 'num_features' predictions for each sample in validation data
#     indices = np.argsort(predictions, axis=1)[:, -num_features:]
#     # Get the predicted numbers using these indices from validation data
#     predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
#     return predicted_numbers
#
#
# def get_numbers(lottery_type):
#     train_data, val_data, max_value, last_data = load_data(lottery_type)
#
#     num_features = train_data.shape[1]
#
#     model = create_model(num_features, max_value)
#
#     train_model(model, train_data, val_data)
#
#     predicted_numbers = predict_numbers(model, val_data, num_features)
#
#     # Convert the last row of predictions to a list and then to JSON
#     predicted_numbers_json = json.dumps(predicted_numbers[-1].tolist())
#
#     return predicted_numbers_json