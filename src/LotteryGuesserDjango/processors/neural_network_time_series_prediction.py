# neural_network_time_series_prediction.py

import os
import numpy as np
import random
from typing import List, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import tensorflow as tf

# Suppress TensorFlow warnings and set logging level to ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a neural network time series prediction model.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        num_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            num_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    num_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Generates a set of lottery numbers using neural network time series prediction.
    """
    # Retrieve past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(numbers_field, flat=True)

    past_draws = list(past_draws_queryset)

    # Filter valid draws
    past_draws = [
        draw for draw in past_draws
        if isinstance(draw, list) and len(draw) == num_numbers
    ]

    if len(past_draws) < 50:
        selected_numbers = random.sample(range(min_num, max_num + 1), num_numbers)
        selected_numbers = [int(num) for num in selected_numbers]
        selected_numbers.sort()
        return selected_numbers

    # Prepare data
    data = np.array([list(map(int, draw)) for draw in past_draws])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    window_size = 10
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(scaled_data[i + window_size])
    X, y = np.array(X), np.array(y)

    # Build and train the model
    model = Sequential([
        Input(shape=(window_size, num_numbers)),
        LSTM(50, activation='relu'),
        Dense(num_numbers)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Prediction
    last_sequence = scaled_data[-window_size:].reshape(1, window_size, num_numbers)
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_numbers = scaler.inverse_transform(predicted_scaled).round().astype(int)[0]

    # Clip to valid range and convert to standard Python int
    predicted_numbers = np.clip(predicted_numbers, min_num, max_num).astype(int)

    # Ensure unique numbers
    predicted_numbers = set(predicted_numbers)

    # Fill missing numbers if needed
    while len(predicted_numbers) < num_numbers:
        predicted_numbers.add(random.randint(min_num, max_num))

    # Return sorted list of numbers
    return sorted([int(num) for num in predicted_numbers])[:num_numbers]
