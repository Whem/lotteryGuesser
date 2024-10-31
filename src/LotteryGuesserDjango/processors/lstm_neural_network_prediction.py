# lstm_neural_network_prediction.py

import numpy as np
from typing import List, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
import os

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using LSTM neural networks for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        required_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        is_main=True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            required_numbers=int(lottery_type_instance.additional_numbers_count),
            is_main=False
        )

    return main_numbers, additional_numbers

def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """
    Generate a set of numbers using LSTM analysis.
    """
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return random_selection(min_num, max_num, required_numbers)

    # Prepare and train LSTM model
    model = train_lstm_model(past_draws, required_numbers)

    # Generate predictions
    predicted_numbers = generate_predictions(
        model,
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)

def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """
    Get historical lottery data.
    """
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id'))

    if is_main:
        draws = [draw.lottery_type_number for draw in past_draws
                 if isinstance(draw.lottery_type_number, list)]
    else:
        draws = [draw.additional_numbers for draw in past_draws
                 if hasattr(draw, 'additional_numbers') and
                 isinstance(draw.additional_numbers, list)]

    return [[int(num) for num in draw] for draw in draws]

def prepare_sequences(past_draws: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.
    """
    sequences = np.array(past_draws)
    X = sequences[:-1]
    y = sequences[1:]

    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1]))

    return X, y

def build_lstm_model(input_shape: Tuple[int, int], output_size: int) -> Sequential:
    """
    Build LSTM model architecture.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(past_draws: List[List[int]], required_numbers: int) -> Sequential:
    """
    Train LSTM model on historical data.
    """
    X, y = prepare_sequences(past_draws)

    model = build_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        output_size=required_numbers
    )

    # Suppress training progress output to prevent UnicodeEncodeError
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model.fit(X, y, epochs=50, verbose=0)
    return model

def generate_predictions(
        model: Sequential,
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """
    Generate predictions using trained LSTM model.
    """
    # Prepare last sequence for prediction
    last_sequence = np.array(past_draws[-1])
    last_sequence = last_sequence.reshape((1, len(last_sequence), 1))

    # Get raw predictions
    predictions = model.predict(last_sequence, verbose=0)
    predictions = predictions.flatten()

    # Process predictions
    predicted_numbers = process_predictions(
        predictions,
        min_num,
        max_num,
        required_numbers
    )

    return predicted_numbers

def process_predictions(
        predictions: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """
    Process raw predictions into valid lottery numbers.
    """
    # Round and constrain to valid range
    numbers = [int(round(num)) for num in predictions]
    numbers = [max(min_num, min(num, max_num)) for num in numbers]

    # Ensure uniqueness
    numbers = list(set(numbers))

    # Fill if needed
    if len(numbers) < required_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(numbers)
        numbers.extend(random.sample(
            list(remaining),
            required_numbers - len(numbers)
        ))

    # Ensure numbers are standard Python int
    numbers = [int(num) for num in numbers]

    return numbers[:required_numbers]

def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """
    Generate random number selection.
    """
    return sorted(random.sample(
        range(min_num, max_num + 1),
        count
    ))
