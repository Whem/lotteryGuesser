# ensemble_random_forest_prediction.py
import random
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Random Forest ensemble predictor for combined lottery types.
    Returns (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using Random Forest ensemble."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 10:
        return random_number_set(min_num, max_num, required_numbers)

    # Prepare training data
    X, y = prepare_training_data(past_draws, min_num, max_num)

    # Train model and get predictions
    predicted_numbers = train_and_predict(
        X, y,
        past_draws[-1],
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
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

    # Ensure consistent length
    required_length = len(draws[0]) if draws else 0
    return [draw for draw in draws if len(draw) == required_length]


def prepare_training_data(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data for the Random Forest model."""
    X = []
    y = []

    # Create input-output pairs
    for i in range(len(past_draws) - 1):
        X.append(past_draws[i])

        # Create one-hot encoded target
        target = [0] * (max_num - min_num + 1)
        for num in past_draws[i + 1]:
            if min_num <= num <= max_num:
                target[num - min_num] = 1
        y.append(target)

    return np.array(X), np.array(y)


def train_and_predict(
        X: np.ndarray,
        y: np.ndarray,
        last_draw: List[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Train Random Forest model and generate predictions."""
    try:
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X, y)

        # Get prediction
        prediction = model.predict([last_draw])[0]

        # Extract predicted numbers
        predicted_numbers = [
            i + min_num
            for i, val in enumerate(prediction)
            if val == 1
        ]

        # Handle prediction results
        if len(predicted_numbers) >= required_numbers:
            return predicted_numbers[:required_numbers]
        else:
            return fill_remaining_numbers(
                predicted_numbers,
                min_num,
                max_num,
                required_numbers
            )

    except Exception as e:
        print(f"Error in Random Forest prediction: {str(e)}")
        return random_number_set(min_num, max_num, required_numbers)


def fill_remaining_numbers(
        predicted_numbers: List[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Fill missing numbers with random selection."""
    numbers = set(predicted_numbers)
    all_numbers = set(range(min_num, max_num + 1))

    # Get remaining available numbers
    remaining = list(all_numbers - numbers)
    random.shuffle(remaining)

    # Add needed numbers
    numbers.update(remaining[:required_numbers - len(numbers)])

    return sorted(list(numbers))[:required_numbers]


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))