# decision_tree_prediction.py
import numpy as np
import random
from typing import List, Tuple, Set
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Decision tree based predictor for combined lottery types.
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
        is_main: bool,
        window_size: int = 5
) -> List[int]:
    """Generate predictions using decision tree analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < window_size + 1:
        return random_number_set(min_num, max_num, required_numbers)

    # Prepare training data
    X, y = prepare_training_data(past_draws, window_size, max_num)

    if len(X) == 0 or len(y) == 0:
        return random_number_set(min_num, max_num, required_numbers)

    # Train model and get predictions
    predicted_numbers = train_and_predict(
        X, y,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data based on number type."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:200])

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def prepare_training_data(
        past_draws: List[List[int]],
        window_size: int,
        max_num: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data for the decision tree."""
    X = []
    y = []

    for i in range(len(past_draws) - window_size):
        # Create feature window
        window = past_draws[i:i + window_size]
        flat_window = [num for draw in window for num in draw]
        X.append(flat_window)

        # Create target labels
        target = [0] * (max_num + 1)
        for num in past_draws[i + window_size]:
            if 0 <= num <= max_num:
                target[num] = 1
        y.append(target)

    return np.array(X), np.array(y)


def train_and_predict(
        X: np.ndarray,
        y: np.ndarray,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Train decision tree model and generate predictions."""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=0.2,
            random_state=42
        )

        # Train model
        base_model = DecisionTreeClassifier(random_state=42)
        model = MultiOutputClassifier(base_model)
        model.fit(X_train, y_train)

        # Get predictions
        predictions = model.predict(X_test)

        # Extract predicted numbers
        predicted_numbers = set()
        for pred in predictions:
            nums = [i for i, val in enumerate(pred) if val == 1]
            predicted_numbers.update(nums)
            if len(predicted_numbers) >= required_numbers:
                break

        # Filter and fill numbers
        predicted_numbers = filter_and_fill_numbers(
            predicted_numbers,
            min_num,
            max_num,
            required_numbers
        )

        return predicted_numbers

    except Exception as e:
        print(f"Error in training/prediction: {str(e)}")
        return random_number_set(min_num, max_num, required_numbers)


def filter_and_fill_numbers(
        numbers: Set[int],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Filter invalid numbers and fill missing ones."""
    # Filter valid numbers
    valid_numbers = {
        num for num in numbers
        if min_num <= num <= max_num
    }

    # Fill missing numbers
    while len(valid_numbers) < required_numbers:
        new_number = random.randint(min_num, max_num)
        valid_numbers.add(new_number)

    return sorted(list(valid_numbers)[:required_numbers])


def random_number_set(min_num: int, max_num: int, required_numbers: int) -> List[int]:
    """Generate a random set of numbers."""
    return sorted(random.sample(range(min_num, max_num + 1), required_numbers))