import numpy as np
import random
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Fetch past draws and convert to list
    past_draws = list(
        lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id').values_list('lottery_type_number', flat=True)
    )

    # Ensure all draws have the correct number of elements
    correct_length = lottery_type_instance.pieces_of_draw_numbers
    past_draws = [draw for draw in past_draws if len(draw) == correct_length]

    if not past_draws:
        raise ValueError("No valid past draws found.")

    # Prepare training data
    X = []
    y = []
    window_size = 5  # Example window size

    for i in range(len(past_draws) - window_size):
        window = past_draws[i:i + window_size]
        X.append(np.concatenate(window))
        y.append(past_draws[i + window_size])

    X = np.array(X)
    y = np.array(y)

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Multi-label encoding
    y_multi = np.zeros((len(y), lottery_type_instance.max_number + 1))
    for i, draw in enumerate(y):
        y_multi[i, draw] = 1

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_multi, test_size=0.2, random_state=42
    )

    # Create and train KNN model with MultiOutputClassifier
    base_model = KNeighborsClassifier(n_neighbors=5)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Select numbers based on highest probabilities
    predicted_numbers = set()
    for pred in predictions:
        nums = np.argsort(pred)[-correct_length:]
        predicted_numbers.update(nums.tolist())  # Convert to standard Python list
        if len(predicted_numbers) >= correct_length:
            break

    # Fill missing numbers randomly if needed
    while len(predicted_numbers) < correct_length:
        new_number = random.randint(
            lottery_type_instance.min_number, lottery_type_instance.max_number
        )
        predicted_numbers.add(new_number)

    # Ensure numbers are within the specified range
    predicted_numbers = [
        int(num) for num in predicted_numbers  # Convert to standard Python int
        if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number
    ]

    return sorted(predicted_numbers)[:correct_length]