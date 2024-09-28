# time_series_arima_prediction.py

import random
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using ARIMA time series prediction.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    import warnings
    warnings.filterwarnings("ignore")

    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_numbers = []
    for draw in past_draws_queryset:
        if isinstance(draw, list):
            past_numbers.extend(draw)

    if len(past_numbers) < 20:
        # Not enough data to train the model
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Prepare the data
    df = pd.DataFrame({'numbers': past_numbers})

    # Fit the ARIMA model
    model = ARIMA(df['numbers'], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast future numbers
    forecast = model_fit.forecast(steps=total_numbers * 2)
    predicted_numbers = forecast.round().astype(int).tolist()

    # Filter numbers within the valid range and remove duplicates
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
    predicted_numbers = list(set(predicted_numbers))

    # Ensure we have enough numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    else:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the numbers
    predicted_numbers.sort()
    return predicted_numbers
