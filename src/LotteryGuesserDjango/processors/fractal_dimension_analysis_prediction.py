# fractal_dimension_analysis_prediction.py
import numpy as np
from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fractal dimension predictor for combined lottery types.
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
    """Generate numbers using fractal dimension analysis."""
    # Get historical data
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return list(range(min_num, min_num + required_numbers))

    # Convert to time series
    time_series = np.array(past_draws).flatten()

    # Calculate fractal dimension
    fractal_dimension = calculate_higuchi_dimension(time_series)

    # Generate predictions
    predicted_numbers = generate_predictions(
        time_series,
        fractal_dimension,
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

    return [[int(num) for num in draw] for draw in draws]


def calculate_higuchi_dimension(time_series: np.ndarray, max_k: int = 10) -> float:
    """Calculate Higuchi fractal dimension."""
    try:
        N = len(time_series)
        Lk = np.zeros(max_k)

        for k in range(1, max_k + 1):
            Lm = []

            for m in range(k):
                Lmk = 0
                n_max = int(np.floor((N - m) / k))

                if n_max >= 2:
                    for i in range(1, n_max):
                        Lmk += abs(time_series[m + i * k] - time_series[m + (i - 1) * k])

                    norm = (N - 1) / (k * n_max * k)
                    Lmk = Lmk * norm
                    Lm.append(Lmk)

            if Lm:
                Lk[k - 1] = np.mean(Lm)

        # Calculate fractal dimension through linear regression
        positive_indices = Lk > 0
        if np.sum(positive_indices) >= 2:
            ln_Lk = np.log(Lk[positive_indices])
            ln_k = np.log(np.arange(1, max_k + 1)[positive_indices])
            coeffs = np.polyfit(ln_k, ln_Lk, 1)
            return abs(coeffs[0])  # Return absolute value for stability

    except Exception as e:
        print(f"Error calculating fractal dimension: {str(e)}")

    return 1.0  # Default dimension if calculation fails


def generate_predictions(
        time_series: np.ndarray,
        fractal_dimension: float,
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Generate predictions based on fractal analysis."""
    try:
        # Calculate frequency-based scores
        number_counts = {}
        for number in time_series:
            number = int(number)
            if min_num <= number <= max_num:
                number_counts[number] = number_counts.get(number, 0) + 1

        # Adjust scores using fractal dimension
        adjusted_scores = {
            number: count * fractal_dimension
            for number, count in number_counts.items()
        }

        # Sort numbers by adjusted scores
        sorted_numbers = sorted(
            adjusted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select unique valid numbers
        predicted_numbers = []
        used_numbers = set()

        for number, _ in sorted_numbers:
            if (min_num <= number <= max_num and
                    number not in used_numbers):
                predicted_numbers.append(number)
                used_numbers.add(number)

            if len(predicted_numbers) >= required_numbers:
                break

        # Fill remaining if needed
        while len(predicted_numbers) < required_numbers:
            for number in range(min_num, max_num + 1):
                if number not in used_numbers:
                    predicted_numbers.append(number)
                    used_numbers.add(number)
                    break

        return predicted_numbers[:required_numbers]

    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return list(range(min_num, min_num + required_numbers))


def get_fractal_statistics(past_draws: List[List[int]]) -> Dict:
    """
    Get comprehensive fractal analysis statistics.

    Returns:
    - fractal_dimension: overall fractal dimension
    - dimension_by_position: fractal dimension for each position
    - complexity_metrics: additional complexity measures
    """
    if not past_draws:
        return {}

    try:
        # Convert to array for analysis
        draw_matrix = np.array(past_draws)
        time_series = draw_matrix.flatten()

        # Calculate overall dimension
        overall_dimension = calculate_higuchi_dimension(time_series)

        # Calculate dimension by position
        position_dimensions = []
        for i in range(draw_matrix.shape[1]):
            dimension = calculate_higuchi_dimension(draw_matrix[:, i])
            position_dimensions.append(dimension)

        stats = {
            'fractal_dimension': float(overall_dimension),
            'dimension_by_position': [float(d) for d in position_dimensions],
            'complexity_metrics': {
                'mean_dimension': float(np.mean(position_dimensions)),
                'dimension_variance': float(np.var(position_dimensions))
            }
        }

        return stats

    except Exception as e:
        print(f"Error calculating fractal statistics: {str(e)}")
        return {}