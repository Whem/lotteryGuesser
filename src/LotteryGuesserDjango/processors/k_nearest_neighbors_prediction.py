#k_nearest_neighbors_prediction.py
import numpy as np
import random
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    # Fetch past draws and convert to list
    past_draws = list(
        lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id')
    )

    # Check if there are enough past draws
    if not past_draws:
        raise ValueError("No valid past draws found.")

    # Prepare data for main numbers
    main_numbers = generate_numbers(
        past_draws,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        lottery_type_instance,
        is_main=True
    )

    # Prepare data for additional numbers if applicable
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            past_draws,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            lottery_type_instance,
            is_main=False
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    past_draws: List[lg_lottery_winner_number],
    min_num: int,
    max_num: int,
    num_numbers: int,
    lottery_type_instance: lg_lottery_type,
    is_main: bool
) -> List[int]:
    # Extract numbers based on whether they are main or additional
    correct_length = num_numbers
    numbers_list = []
    for draw in past_draws:
        if is_main:
            numbers = getattr(draw, 'lottery_type_number', [])
        else:
            numbers = getattr(draw, 'additional_numbers', [])
        if isinstance(numbers, list) and len(numbers) == correct_length:
            numbers_list.append(numbers)

    # Ensure there are enough valid past draws
    if len(numbers_list) <= 5:
        # If not enough valid past draws, return random numbers
        return random.sample(range(min_num, max_num + 1), num_numbers)

    # Prepare training data
    X = []
    y = []
    window_size = 5  # Example window size

    for i in range(len(numbers_list) - window_size):
        window = numbers_list[i:i + window_size]
        X.append(np.concatenate(window))
        y.append(numbers_list[i + window_size])

    X = np.array(X)
    y = np.array(y)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Multi-label encoding
    max_number = max_num
    y_multi = np.zeros((len(y), max_number + 1))
    for i, draw in enumerate(y):
        for num in draw:
            if min_num <= num <= max_number:
                y_multi[i, num - min_num] = 1  # Adjust index to start from 0

    # Split data for training and testing
    if len(X_scaled) < 2:
        # Not enough data to split; use all data for training
        X_train = X_scaled
        y_train = y_multi
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_multi, test_size=0.2, random_state=42
        )

    # Create and train KNN model with MultiOutputClassifier
    base_model = KNeighborsClassifier(n_neighbors=5)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    # Prepare the input for prediction
    last_windows = numbers_list[-window_size:]
    last_window_flat = np.concatenate(last_windows)
    X_predict = scaler.transform([last_window_flat])

    # Predict
    prediction = model.predict(X_predict)[0]

    # Select numbers based on highest probabilities
    nums = np.argsort(prediction)[-num_numbers:]
    predicted_numbers = set((nums + min_num).tolist())  # Adjust back to original number range

    # Fill missing numbers randomly if needed
    while len(predicted_numbers) < num_numbers:
        new_number = random.randint(min_num, max_num)
        predicted_numbers.add(new_number)

    # Ensure numbers are within the specified range and are standard Python ints
    predicted_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

    return sorted(predicted_numbers)[:num_numbers]
