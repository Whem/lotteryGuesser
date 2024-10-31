# neural_network_prediction.py

import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from typing import List, Tuple

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a neural network prediction model.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Generates lottery numbers using a neural network prediction model.
    """
    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list(numbers_field, flat=True)

    past_draws = [draw for draw in past_draws_queryset if isinstance(draw, list) and len(draw) == total_numbers]

    if len(past_draws) < 10:
        # Not enough data to train the model
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Prepare training data
    X = []
    y = []
    for i in range(len(past_draws) - 1):
        X.append(past_draws[i])
        y.append(past_draws[i + 1])

    X = np.array(X)
    y = np.array(y)

    # Flatten y to create multi-labels
    y_flat = []
    for target in y:
        label = [0] * (max_num - min_num + 1)
        for num in target:
            label[int(num) - min_num] = 1
        y_flat.append(label)
    y_flat = np.array(y_flat)

    # Train the neural network
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y_flat)

    # Use the last draw as input for prediction
    last_draw = past_draws[-1]
    prediction = model.predict([last_draw])[0]

    # Extract predicted numbers
    predicted_numbers = [i + min_num for i, val in enumerate(prediction) if val == 1]

    # If we have enough predicted numbers
    if len(predicted_numbers) >= total_numbers:
        selected_numbers = predicted_numbers[:total_numbers]
    else:
        # Fill the remaining numbers randomly
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        selected_numbers = predicted_numbers + remaining_numbers[:total_numbers - len(predicted_numbers)]

    # Ensure the correct number of numbers
    selected_numbers = selected_numbers[:total_numbers]
    selected_numbers.sort()
    return selected_numbers
