# linear_regression_prediction.py
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generate lottery numbers using statistical features and linear regression for both main and additional numbers.
    Returns a tuple (main_numbers, additional_numbers).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance,
        min_number=int(lottery_type_instance.min_number),
        max_number=int(lottery_type_instance.max_number),
        num_picks=int(lottery_type_instance.pieces_of_draw_numbers),
        numbers_field='lottery_type_number'
    )

    # Generate additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.additional_min_number),
            max_number=int(lottery_type_instance.additional_max_number),
            num_picks=int(lottery_type_instance.additional_numbers_count),
            numbers_field='additional_numbers'
        )

    return sorted(main_numbers), sorted(additional_numbers)

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    num_picks: int,
    numbers_field: str
) -> List[int]:
    """
    Generate numbers using statistical features and linear regression.
    """
    # Retrieve past draws
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id').values_list(numbers_field, flat=True)[:100]

    past_draws = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) > 0
    ]

    if len(past_draws) < 20:
        return sorted(np.random.choice(
            range(min_number, max_number + 1),
            num_picks,
            replace=False
        ).tolist())

    X = []
    y = []
    for i in range(len(past_draws) - 1):
        current_draw = [num for num in past_draws[i] if isinstance(num, (int, float))]
        next_draw = [num for num in past_draws[i+1] if isinstance(num, (int, float))]
        if not current_draw or not next_draw:
            continue
        X.append([
            np.mean(current_draw),
            np.std(current_draw),
            max(current_draw),
            min(current_draw),
            i  # time factor
        ])
        y.append(np.mean(next_draw))

    if not X or not y:
        return sorted(np.random.choice(
            range(min_number, max_number + 1),
            num_picks,
            replace=False
        ).tolist())

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    # Prepare input for prediction
    last_draw = [num for num in past_draws[0] if isinstance(num, (int, float))]
    if not last_draw:
        last_draw = [min_number]

    prediction_input = np.array([[
        np.mean(last_draw),
        np.std(last_draw),
        max(last_draw),
        min(last_draw),
        len(past_draws)  # current time
    ]])

    predicted_mean = model.predict(prediction_input)[0]

    # Generate numbers around the predicted mean
    predicted_numbers = set()
    while len(predicted_numbers) < num_picks:
        number = int(round(np.random.normal(predicted_mean)))
        if min_number <= number <= max_number and number not in predicted_numbers:
            predicted_numbers.add(number)
        # Prevent infinite loop
        if len(predicted_numbers) + (max_number - min_number + 1 - len(predicted_numbers)) < num_picks:
            break

    # Fill missing numbers if needed
    if len(predicted_numbers) < num_picks:
        remaining_numbers = set(range(min_number, max_number + 1)) - predicted_numbers
        predicted_numbers.update(random.sample(remaining_numbers, num_picks - len(predicted_numbers)))

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(num) for num in predicted_numbers]

    return sorted(predicted_numbers)[:num_picks]
