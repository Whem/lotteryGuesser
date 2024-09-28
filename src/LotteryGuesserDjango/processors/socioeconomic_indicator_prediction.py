# socioeconomic_indicator_prediction.py

import numpy as np
from typing import List
from algorithms.models import lg_lottery_type
import requests


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Megjegyzés: Ez egy szimulált API hívás. Valós használathoz megfelelő API-t kell használni.
    def get_economic_indicators():
        # Szimulált gazdasági mutatók
        return {
            'gdp_growth': np.random.uniform(0, 5),
            'unemployment_rate': np.random.uniform(3, 10),
            'inflation_rate': np.random.uniform(0, 5),
            'stock_market_index': np.random.uniform(10000, 30000)
        }

    indicators = get_economic_indicators()

    # Normalizáljuk az indikátorokat
    normalized_indicators = {k: (v - min(indicators.values())) / (max(indicators.values()) - min(indicators.values()))
                             for k, v in indicators.items()}

    # Generáljunk számokat az indikátorok alapján
    predicted_numbers = set()
    for _ in range(lottery_type_instance.pieces_of_draw_numbers * 2):  # Generáljunk több számot, mint szükséges
        weighted_sum = sum(v * np.random.random() for v in normalized_indicators.values())
        number = int(lottery_type_instance.min_number +
                     weighted_sum * (lottery_type_instance.max_number - lottery_type_instance.min_number))
        if lottery_type_instance.min_number <= number <= lottery_type_instance.max_number:
            predicted_numbers.add(number)

    # Ha túl sok számot generáltunk, véletlenszerűen távolítsunk el néhányat
    while len(predicted_numbers) > lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.remove(np.random.choice(list(predicted_numbers)))

    # Ha nem elég számot generáltunk, adjunk hozzá véletlenszerű számokat
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(np.random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number + 1))

    return sorted(predicted_numbers)