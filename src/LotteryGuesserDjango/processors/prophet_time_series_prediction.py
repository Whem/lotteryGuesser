# prophet_time_series_prediction.py

import random
import pandas as pd
from prophet import Prophet
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import datetime
from typing import List, Tuple

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using Prophet time series forecasting.
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

    return main_numbers, additional_numbers

def generate_numbers(
    lottery_type_instance: lg_lottery_type,
    min_num: int,
    max_num: int,
    total_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Generate numbers using Prophet time series forecasting.
    """
    # Retrieve past winning numbers with dates
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('lottery_type_number_year', 'lottery_type_number_week').values_list(
        'lottery_type_number_year', 'lottery_type_number_week', numbers_field
    )

    data = []
    for year, week, draw in past_draws_queryset:
        if isinstance(draw, list) and len(draw) > 0:
            try:
                # Approximate date from year and week
                date = datetime.datetime.strptime(f'{year}-W{int(week)}-1', "%Y-W%W-%w").date()
            except ValueError:
                # If parsing fails, skip this entry
                continue
            for num in draw:
                try:
                    num = int(num)
                    data.append({'ds': date, 'y': num})
                except (ValueError, TypeError):
                    continue

    if len(data) < 20:
        # Not enough data
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers = [int(num) for num in selected_numbers]
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

    # Ensure unique numbers and correct count
    predicted_numbers = list(set(predicted_numbers))
    if len(predicted_numbers) < total_numbers:
        remaining_numbers = list(set(range(min_num, max_num + 1)) - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    else:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Convert numbers to standard Python int and sort
    predicted_numbers = [int(num) for num in predicted_numbers]
    predicted_numbers.sort()

    return predicted_numbers
