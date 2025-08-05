# symbolic_regression_prediction.py

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from typing import List, Tuple, Set, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random
from collections import defaultdict
from statistics import mean, stdev  # Added missing imports


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Symbolic Regression (SR) analysis.
    Returns a single list containing both main and additional numbers.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A list containing predicted numbers (main numbers followed by additional numbers if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    # Return combined list
    return main_numbers ,additional_numbers


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        number_field: str,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using Symbolic Regression (SR).
    """
    try:
        # Retrieve past winning numbers
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id')

        past_draws = []
        for draw in past_draws_queryset:
            numbers = getattr(draw, number_field, None)
            if isinstance(numbers, list) and len(numbers) == total_numbers:
                try:
                    past_draws.append([int(num) for num in numbers])
                except (ValueError, TypeError):
                    continue

        if len(past_draws) < 20:
            return generate_random_numbers(min_num, max_num, total_numbers)

        # Prepare features and targets for Symbolic Regression
        window_size = 5  # Number of past draws to consider for prediction
        X, y = create_features_and_targets(past_draws, window_size, total_numbers)

        if X.size == 0 or y.size == 0:
            return generate_random_numbers(min_num, max_num, total_numbers)

        predicted_numbers = []

        # Train and predict for each number position
        for pos in range(total_numbers):
            try:
                # Try using Symbolic Regressor with version compatibility fix
                try:
                    # Define a custom fitness function (Mean Squared Error)
                    mse = make_fitness(
                        function=lambda y, y_pred, _: np.mean((y - y_pred) ** 2),
                        greater_is_better=False
                    )

                    # Initialize Symbolic Regressor with simplified settings
                    est = SymbolicRegressor(
                        population_size=100,  # Reduced for faster execution
                        generations=10,       # Reduced for faster execution
                        tournament_size=10,
                        stopping_criteria=0.01,
                        const_range=(0, 10),
                        init_depth=(2, 4),
                        function_set=['add', 'sub', 'mul'],  # Simplified function set
                        metric='mse',  # Use string instead of custom metric
                        parsimony_coefficient=0.001,
                        max_samples=0.8,
                        verbose=0,
                        n_jobs=1,
                        random_state=pos  # Different seed for each position
                    )

                    # Fit the model
                    est.fit(X, y[:, pos])

                    # Predict the next number using the last window
                    last_window = past_draws[-window_size:]
                    flattened_last_window = np.array([num for draw in last_window for num in draw]).reshape(1, -1)
                    predicted_value = est.predict(flattened_last_window)[0]

                    # Round and clamp the predicted number within the valid range
                    predicted_number = int(round(predicted_value))
                    predicted_number = max(min_num, min(max_num, predicted_number))

                    predicted_numbers.append(predicted_number)

                except AttributeError as attr_err:
                    if '_validate_data' in str(attr_err):
                        # Version compatibility issue - use simple linear trend instead
                        position_values = y[:, pos]
                        if len(position_values) >= 2:
                            trend = position_values[-1] - position_values[-2]
                            predicted_number = int(position_values[-1] + trend)
                            predicted_number = max(min_num, min(max_num, predicted_number))
                            predicted_numbers.append(predicted_number)
                    else:
                        raise attr_err

            except Exception as e:
                # Fallback: use mean of recent draws for this position
                try:
                    recent_values = [draw[pos] for draw in past_draws[-10:] if len(draw) > pos]
                    if recent_values:
                        predicted_number = int(np.mean(recent_values))
                        predicted_number = max(min_num, min(max_num, predicted_number))
                        predicted_numbers.append(predicted_number)
                except:
                    pass  # Skip this position if all fails
                continue

        # Ensure uniqueness
        predicted_numbers = list(dict.fromkeys(predicted_numbers))

        # If not enough unique numbers, fill with weighted random choices
        if len(predicted_numbers) < total_numbers:
            weights = calculate_field_weights(
                past_draws=past_draws,
                min_num=min_num,
                max_num=max_num,
                excluded_numbers=set(predicted_numbers)
            )

            while len(predicted_numbers) < total_numbers:
                number = weighted_random_choice(
                    weights,
                    set(range(min_num, max_num + 1)) - set(predicted_numbers)
                )
                if number is not None:
                    predicted_numbers.append(number)

        # Sort and trim to the required number of numbers
        return sorted(predicted_numbers)[:total_numbers]

    except Exception as e:
        print(f"Error in generate_numbers: {str(e)}")
        return generate_random_numbers(min_num, max_num, total_numbers)


def create_features_and_targets(past_draws: List[List[int]], window_size: int, total_numbers: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """Creates feature and target datasets for Symbolic Regression."""
    try:
        X = []
        y = []

        for i in range(len(past_draws) - window_size):
            window = past_draws[i:i + window_size]
            next_draw = past_draws[i + window_size]
            # Flatten the window into a single feature vector
            flattened_window = [num for draw in window for num in draw]
            X.append(flattened_window)
            y.append(next_draw)

        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error in create_features_and_targets: {str(e)}")
        return np.array([]), np.array([])


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """Calculate weights for field selection."""
    weights = defaultdict(float)

    try:
        if not past_draws:
            return {num: 1.0 for num in range(min_num, max_num + 1)
                    if num not in excluded_numbers}

        # Calculate statistics
        all_numbers = [num for draw in past_draws for num in draw]
        if not all_numbers:
            return {num: 1.0 for num in range(min_num, max_num + 1)
                    if num not in excluded_numbers}

        mean_val = mean(all_numbers)
        std_dev = stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        # Calculate weights using z-score
        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            z_score = abs(num - mean_val) / std_dev if std_dev != 0 else 0.0
            weights[num] = max(0, 1 - (z_score / 3))  # Normalize z-score effect

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

    except Exception as e:
        print(f"Weight calculation error: {str(e)}")
        # Fallback to uniform weights
        weights = {num: 1.0 for num in range(min_num, max_num + 1)
                   if num not in excluded_numbers}

    return weights


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """Select a random number based on weights."""
    try:
        if not available_numbers:
            return None

        numbers = list(available_numbers)
        if not numbers:
            return None

        weights_list = [weights.get(num, 0.0) for num in numbers]
        total_weight = sum(weights_list)

        if total_weight <= 0:
            return random.choice(numbers)

        weights_list = [w / total_weight for w in weights_list]
        return random.choices(numbers, weights=weights_list, k=1)[0]

    except Exception as e:
        print(f"Error in weighted_random_choice: {str(e)}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(min_num: int, max_num: int, total_numbers: int) -> List[int]:
    """Generate random numbers safely."""
    try:
        if max_num < min_num or total_numbers <= 0:
            return []

        numbers = set()
        available_range = list(range(min_num, max_num + 1))

        if total_numbers > len(available_range):
            total_numbers = len(available_range)

        while len(numbers) < total_numbers:
            numbers.add(random.choice(available_range))

        return sorted(list(numbers))
    except Exception as e:
        print(f"Error in generate_random_numbers: {str(e)}")
        return list(range(min_num, min(min_num + total_numbers, max_num + 1)))