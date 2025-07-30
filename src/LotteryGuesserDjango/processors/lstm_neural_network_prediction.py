# lstm_neural_network_prediction.py
"""
Fejlesztett LSTM neurális hálózat lottószám predikció
Speciálisan az Eurojackpot-hoz optimalizálva
"""

import numpy as np
from typing import List, Tuple, Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Build advanced LSTM model architecture with modern improvements.
    """
    model = Sequential([
        # Bidirectional LSTM layers for better pattern recognition
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), 
                     input_shape=input_shape),
        BatchNormalization(),
        
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        BatchNormalization(),
        
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        BatchNormalization(),
        
        # Dense layers with regularization
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(output_size, activation='linear')
    ])
    
    # Advanced optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def train_lstm_model(past_draws: List[List[int]], required_numbers: int) -> Sequential:
    """
    Train advanced LSTM model on historical data with improved techniques.
    """
    X, y = prepare_sequences(past_draws)
    
    # Data normalization for better training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    y_reshaped = y.reshape(-1, y.shape[-1])
    y_scaled = scaler_y.fit_transform(y_reshaped)
    y_scaled = y_scaled.reshape(y.shape)

    model = build_lstm_model(
        input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
        output_size=required_numbers
    )

    # Advanced callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=0.0001,
        verbose=0
    )

    # Suppress training progress output to prevent UnicodeEncodeError
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Train with validation split and callbacks
    history = model.fit(
        X_scaled, y_scaled,
        epochs=100,
        batch_size=min(32, X_scaled.shape[0] // 4),
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Store scalers for later use in prediction
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    
    logger.info(f"LSTM model trained with {len(history.history['loss'])} epochs")
    
    return model

def generate_predictions(
        model: Sequential,
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """
    Generate predictions using trained LSTM model with proper scaling.
    """
    # Prepare multiple recent sequences for ensemble prediction
    sequences_to_use = min(5, len(past_draws))
    ensemble_predictions = []
    
    for i in range(sequences_to_use):
        # Prepare sequence for prediction
        sequence_idx = -(i + 1)
        last_sequence = np.array(past_draws[sequence_idx])
        last_sequence = last_sequence.reshape((1, len(last_sequence), 1))
        
        # Apply scaling if available
        if hasattr(model, 'scaler_X'):
            last_sequence_reshaped = last_sequence.reshape(-1, last_sequence.shape[-1])
            last_sequence_scaled = model.scaler_X.transform(last_sequence_reshaped)
            last_sequence = last_sequence_scaled.reshape(last_sequence.shape)

        # Get raw predictions
        raw_predictions = model.predict(last_sequence, verbose=0)
        
        # Apply inverse scaling if available
        if hasattr(model, 'scaler_y'):
            predictions_reshaped = raw_predictions.reshape(-1, raw_predictions.shape[-1])
            predictions = model.scaler_y.inverse_transform(predictions_reshaped)
            predictions = predictions.reshape(raw_predictions.shape)
        else:
            predictions = raw_predictions
            
        ensemble_predictions.append(predictions.flatten())
    
    # Average ensemble predictions for stability
    final_predictions = np.mean(ensemble_predictions, axis=0)

    # Process predictions with advanced logic
    predicted_numbers = process_advanced_predictions(
        final_predictions,
        min_num,
        max_num,
        required_numbers,
        past_draws
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


def process_advanced_predictions(
        predictions: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int,
        past_draws: List[List[int]]
) -> List[int]:
    """
    Advanced processing of predictions with statistical validation.
    """
    # Round and constrain to valid range
    initial_numbers = [int(round(num)) for num in predictions]
    initial_numbers = [max(min_num, min(num, max_num)) for num in initial_numbers]
    
    # Remove duplicates while preserving order
    seen = set()
    numbers = []
    for num in initial_numbers:
        if num not in seen:
            numbers.append(num)
            seen.add(num)
    
    # Calculate frequency weights from recent draws
    freq_weights = calculate_frequency_weights(past_draws, min_num, max_num)
    
    # If we need more numbers, use frequency-weighted selection
    if len(numbers) < required_numbers:
        remaining_pool = [(num, weight) for num, weight in freq_weights.items() 
                         if num not in numbers]
        remaining_pool.sort(key=lambda x: x[1], reverse=True)
        
        # Add high-frequency numbers first
        for num, _ in remaining_pool:
            if len(numbers) >= required_numbers:
                break
            numbers.append(num)
    
    # Final fallback with random selection if still needed
    if len(numbers) < required_numbers:
        remaining = set(range(min_num, max_num + 1)) - set(numbers)
        numbers.extend(random.sample(
            list(remaining),
            required_numbers - len(numbers)
        ))

    # Ensure numbers are standard Python int and return required amount
    numbers = [int(num) for num in numbers[:required_numbers]]
    
    return numbers


def calculate_frequency_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Dict[int, float]:
    """
    Calculate frequency weights with recency bias.
    """
    freq_weights = {i: 0.0 for i in range(min_num, max_num + 1)}
    
    # Weight recent draws more heavily
    for i, draw in enumerate(past_draws):
        weight = 1.0 + (i / len(past_draws))  # More recent = higher weight
        for num in draw:
            if min_num <= num <= max_num:
                freq_weights[num] += weight
    
    # Normalize weights
    max_weight = max(freq_weights.values()) if freq_weights.values() else 1.0
    if max_weight > 0:
        freq_weights = {k: v / max_weight for k, v in freq_weights.items()}
    
    return freq_weights

def random_selection(min_num: int, max_num: int, count: int) -> List[int]:
    """
    Generate random number selection.
    """
    return sorted(random.sample(
        range(min_num, max_num + 1),
        count
    ))
