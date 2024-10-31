# socioeconomic_indicator_prediction.py

import random
import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a simple reinforcement learning approach based on socioeconomic indicators.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    # Generate main numbers
    main_numbers = generate_numbers(
        lottery_type_instance=lottery_type_instance,
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_numbers(
            lottery_type_instance=lottery_type_instance,
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    return main_numbers, additional_numbers


def generate_numbers(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        total_numbers: int
) -> List[int]:
    """
    Generates a list of lottery numbers using a simple reinforcement learning approach based on socioeconomic indicators.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of predicted lottery numbers.
    """

    # Note: Reinforcement learning typically requires a defined environment and reward system.
    # For the purpose of this example, we'll use a simplified approach.

    def get_economic_indicators():
        """
        Simulates fetching socioeconomic indicators.
        In a real-world scenario, replace this with actual API calls to retrieve real data.

        Returns:
            dict: A dictionary containing simulated economic indicators.
        """
        # Simulated socioeconomic indicators
        return {
            'gdp_growth': np.random.uniform(0, 5),
            'unemployment_rate': np.random.uniform(3, 10),
            'inflation_rate': np.random.uniform(0, 5),
            'stock_market_index': np.random.uniform(10000, 30000)
        }

    indicators = get_economic_indicators()

    # Normalize the indicators
    indicator_values = list(indicators.values())
    min_val = min(indicator_values)
    max_val = max(indicator_values)
    if max_val - min_val == 0:
        normalized_indicators = {k: 0.5 for k in indicators.keys()}  # Avoid division by zero
    else:
        normalized_indicators = {
            k: (v - min_val) / (max_val - min_val) for k, v in indicators.items()
        }

    # Generate numbers based on indicators
    predicted_numbers = set()
    for _ in range(total_numbers * 2):  # Generate more numbers than needed to increase uniqueness
        weighted_sum = sum(v * np.random.random() for v in normalized_indicators.values())
        number = int(min_num + weighted_sum * (max_num - min_num))
        if min_num <= number <= max_num:
            predicted_numbers.add(number)

    # If too many numbers were generated, randomly remove some
    while len(predicted_numbers) > total_numbers:
        predicted_numbers.remove(random.choice(list(predicted_numbers)))

    # If not enough numbers, fill with random numbers
    while len(predicted_numbers) < total_numbers:
        predicted_numbers.add(random.randint(min_num, max_num))

    return sorted(predicted_numbers)
