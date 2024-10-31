# kalman_filter_prediction.py

# Generates lottery numbers using the Kalman Filter approach.
# Applies a Kalman filter to predict the next number in each position based on past draws.
# Supports both main and additional numbers.

import numpy as np
from pykalman import KalmanFilter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance):
    """
    Generate lottery numbers using Kalman Filter.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type.

    Returns:
    - A tuple containing two lists: (main_numbers, additional_numbers).
    """

    # Main numbers configuration
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    # Process main numbers
    past_main_numbers = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_main_numbers) < 10:
        # If insufficient data, return the first 'total_numbers' numbers
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers, []

    # Convert past draws to a numpy matrix
    draw_matrix = np.array(past_main_numbers)

    # Initialize predicted numbers list
    predicted_numbers = []

    # Apply Kalman Filter to each position
    for i in range(total_numbers):
        # Select the time series for the current position
        series = draw_matrix[:, i]

        # Initialize Kalman Filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=series[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01
        )

        # Filtering and prediction
        state_means, _ = kf.filter(series)
        next_state_mean, _ = kf.filter_update(
            filtered_state_mean=state_means[-1],
            filtered_state_covariance=1,
            observation=None
        )

        predicted_number = int(round(next_state_mean[0]))

        # Adjust to range
        if predicted_number < min_num:
            predicted_number = min_num
        elif predicted_number > max_num:
            predicted_number = max_num

        predicted_numbers.append(predicted_number)

    # Remove duplicates
    predicted_numbers = list(set(predicted_numbers))

    # If we have fewer numbers than needed, fill with the most common numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_main_numbers for number in draw]
        number_counts = {}
        for number in all_numbers:
            number_counts[number] = number_counts.get(number, 0) + 1
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_numbers:
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Sort and trim to required length
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()

    # Ensure all numbers are standard Python int
    predicted_numbers = [int(n) for n in predicted_numbers]

    # Handle additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Additional numbers configuration
        additional_min_num = lottery_type_instance.additional_min_number
        additional_max_num = lottery_type_instance.additional_max_number
        additional_total_numbers = lottery_type_instance.additional_numbers_count

        # Process additional numbers from past draws
        past_draws_additional_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id').values_list('additional_numbers', flat=True)

        past_additional_numbers = [
            draw for draw in past_draws_additional_queryset
            if isinstance(draw, list) and len(draw) == additional_total_numbers
        ]

        if len(past_additional_numbers) < 10:
            # If insufficient data, return first 'additional_total_numbers' numbers
            additional_numbers = list(range(additional_min_num, additional_min_num + additional_total_numbers))
        else:
            # Convert past additional draws to numpy matrix
            draw_additional_matrix = np.array(past_additional_numbers)

            # Initialize predicted additional numbers list
            predicted_additional_numbers = []

            # Apply Kalman Filter to each position
            for i in range(additional_total_numbers):
                # Select the time series for the current position
                series = draw_additional_matrix[:, i]

                # Initialize Kalman Filter
                kf = KalmanFilter(
                    transition_matrices=[1],
                    observation_matrices=[1],
                    initial_state_mean=series[0],
                    initial_state_covariance=1,
                    observation_covariance=1,
                    transition_covariance=0.01
                )

                # Filtering and prediction
                state_means, _ = kf.filter(series)
                next_state_mean, _ = kf.filter_update(
                    filtered_state_mean=state_means[-1],
                    filtered_state_covariance=1,
                    observation=None
                )

                predicted_number = int(round(next_state_mean[0]))

                # Adjust to range
                if predicted_number < additional_min_num:
                    predicted_number = additional_min_num
                elif predicted_number > additional_max_num:
                    predicted_number = additional_max_num

                predicted_additional_numbers.append(predicted_number)

            # Remove duplicates
            predicted_additional_numbers = list(set(predicted_additional_numbers))

            # If fewer numbers than needed, fill with the most common additional numbers
            if len(predicted_additional_numbers) < additional_total_numbers:
                all_additional_numbers = [number for draw in past_additional_numbers for number in draw]
                number_counts = {}
                for number in all_additional_numbers:
                    number_counts[number] = number_counts.get(number, 0) + 1
                sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
                for num, _ in sorted_numbers:
                    if num not in predicted_additional_numbers:
                        predicted_additional_numbers.append(num)
                    if len(predicted_additional_numbers) == additional_total_numbers:
                        break

            # Sort and trim to required length
            predicted_additional_numbers = predicted_additional_numbers[:additional_total_numbers]
            predicted_additional_numbers.sort()

            # Ensure all numbers are standard Python int
            predicted_additional_numbers = [int(n) for n in predicted_additional_numbers]

            additional_numbers = predicted_additional_numbers

    return predicted_numbers, additional_numbers
