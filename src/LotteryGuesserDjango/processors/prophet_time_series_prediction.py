# prophet_time_series_prediction.py

import random
import pandas as pd
from prophet import Prophet
from algorithms.models import lg_lottery_winner_number
import datetime

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using Prophet time series forecasting.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers with dates
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('lottery_type_number_year', 'lottery_type_number_week').values_list(
        'lottery_type_number_year', 'lottery_type_number_week', 'lottery_type_number'
    )

    data = []
    for year, week, draw in past_draws_queryset:
        if isinstance(draw, list) and len(draw) == total_numbers:
            # Approximate date from year and week
            date = datetime.datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%W-%w")
            for num in draw:
                data.append({'ds': date, 'y': num})

    if len(data) < 20:
        # Not enough data
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    df = pd.DataFrame(data)

    # Fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Predict future numbers
    future = model.make_future_dataframe(periods=1, freq='W')
    forecast = model.predict(future)

    # Extract predicted numbers
    predicted_numbers = forecast['yhat'].iloc[-total_numbers * 2:].round().astype(int).tolist()
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]
    predicted_numbers = list(set(predicted_numbers))[:total_numbers]
    predicted_numbers.sort()

    return predicted_numbers
