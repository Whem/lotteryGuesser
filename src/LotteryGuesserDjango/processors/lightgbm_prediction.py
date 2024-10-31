# lightgbm_prediction.py

# Generates lottery numbers using LightGBM-based analysis.
# Applies LightGBM regression models to predict each number position based on past draws.
# Supports both main and additional numbers.

import numpy as np
from collections import Counter
import lightgbm as lgb
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance, num_iterations=100, random_state=42):
    """
    Generate lottery numbers using LightGBM-based analysis.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - num_iterations: Number of training iterations for LightGBM.
    - random_state: Random state for model reproducibility.

    Returns:
    - A tuple containing two lists: (main_numbers, additional_numbers).
    """
    # Main numbers configuration
    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id')

    past_draws = []
    for draw_entry in past_draws_queryset:
        draw_numbers = draw_entry.lottery_type_number
        if isinstance(draw_numbers, list) and len(draw_numbers) == total_numbers:
            past_draws.append([int(num) for num in draw_numbers])

    if len(past_draws) < 20:
        # If insufficient data, return the first 'total_numbers' numbers
        main_numbers = list(range(min_num, min_num + total_numbers))
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_min_num = int(lottery_type_instance.additional_min_number)
            additional_total_numbers = int(lottery_type_instance.additional_numbers_count)
            additional_numbers = list(range(additional_min_num, additional_min_num + additional_total_numbers))
        return main_numbers, additional_numbers

    # Prepare data for LightGBM
    X = np.arange(len(past_draws)).reshape(-1, 1)  # Indexes of draws

    # Train models for main numbers
    main_models = []
    for num_position in range(total_numbers):
        y = np.array([draw[num_position] for draw in past_draws])
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'seed': random_state
        }
        model = lgb.train(params, train_data, num_boost_round=num_iterations)
        main_models.append(model)

    # Predict next main numbers
    next_index = np.array([[len(past_draws)]])
    predicted_numbers = []
    for idx, model in enumerate(main_models):
        pred = model.predict(next_index)[0]
        pred_int = int(round(pred))
        pred_int = max(min_num, min(pred_int, max_num))
        if pred_int not in predicted_numbers:
            predicted_numbers.append(pred_int)
        if len(predicted_numbers) == total_numbers:
            break

    # If fewer numbers than needed, fill with the most common numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]
        for num in most_common_numbers:
            predicted_numbers.append(int(num))
            if len(predicted_numbers) == total_numbers:
                break

    # Ensure unique numbers within the allowed range
    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # If still fewer numbers, add deterministically
    if len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(int(num))
                if len(predicted_numbers) == total_numbers:
                    break

    # Final sorting
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()

    main_numbers = predicted_numbers

    # Handle additional numbers if needed
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_min_num = int(lottery_type_instance.additional_min_number)
        additional_max_num = int(lottery_type_instance.additional_max_number)
        additional_total_numbers = int(lottery_type_instance.additional_numbers_count)

        # Retrieve past additional numbers
        past_additional_draws = []
        for draw_entry in past_draws_queryset:
            additional_draw_numbers = getattr(draw_entry, 'additional_numbers', None)
            if isinstance(additional_draw_numbers, list) and len(additional_draw_numbers) == additional_total_numbers:
                past_additional_draws.append([int(num) for num in additional_draw_numbers])

        if len(past_additional_draws) < 20:
            # If insufficient data, return first 'additional_total_numbers' numbers
            additional_numbers = list(range(additional_min_num, additional_min_num + additional_total_numbers))
        else:
            # Prepare data for LightGBM for additional numbers
            X_additional = np.arange(len(past_additional_draws)).reshape(-1, 1)

            # Train models for additional numbers
            additional_models = []
            for num_position in range(additional_total_numbers):
                y_additional = np.array([draw[num_position] for draw in past_additional_draws])
                train_data_additional = lgb.Dataset(X_additional, label=y_additional)
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'seed': random_state
                }
                model = lgb.train(params, train_data_additional, num_boost_round=num_iterations)
                additional_models.append(model)

            # Predict next additional numbers
            next_index_additional = np.array([[len(past_additional_draws)]])
            predicted_additional_numbers = []
            for idx, model in enumerate(additional_models):
                pred = model.predict(next_index_additional)[0]
                pred_int = int(round(pred))
                pred_int = max(additional_min_num, min(pred_int, additional_max_num))
                if pred_int not in predicted_additional_numbers:
                    predicted_additional_numbers.append(pred_int)
                if len(predicted_additional_numbers) == additional_total_numbers:
                    break

            # If fewer numbers than needed, fill with most common additional numbers
            if len(predicted_additional_numbers) < additional_total_numbers:
                all_additional_numbers = [num for draw in past_additional_draws for num in draw]
                number_counts_additional = Counter(all_additional_numbers)
                most_common_additional_numbers = [num for num, count in number_counts_additional.most_common()
                                                  if num not in predicted_additional_numbers]
                for num in most_common_additional_numbers:
                    predicted_additional_numbers.append(int(num))
                    if len(predicted_additional_numbers) == additional_total_numbers:
                        break

            # Ensure unique numbers within the allowed range
            predicted_additional_numbers = [int(num) for num in predicted_additional_numbers
                                            if additional_min_num <= num <= additional_max_num]

            # If still fewer numbers, add deterministically
            if len(predicted_additional_numbers) < additional_total_numbers:
                for num in range(additional_min_num, additional_max_num + 1):
                    if num not in predicted_additional_numbers:
                        predicted_additional_numbers.append(int(num))
                        if len(predicted_additional_numbers) == additional_total_numbers:
                            break

            # Final sorting
            predicted_additional_numbers = predicted_additional_numbers[:additional_total_numbers]
            predicted_additional_numbers.sort()

            additional_numbers = predicted_additional_numbers

    return main_numbers, additional_numbers
