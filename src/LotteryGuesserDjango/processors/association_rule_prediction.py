#association_rule_prediction.py
import numpy as np
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance: lg_lottery_type, min_support=0.05, metric="confidence", min_threshold=0.6) -> Tuple[List[int], List[int]]:
    """
    Generate main and additional numbers based on Association Rule Mining.
    Returns a tuple of (main_numbers, additional_numbers).
    """
    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        selected_numbers = list(range(min_num, min_num + total_numbers))
        additional_numbers = list(range(lottery_type_instance.additional_min_number, lottery_type_instance.additional_min_number + lottery_type_instance.additional_numbers_count))
        return selected_numbers, additional_numbers

    # Convert draws to DataFrame for association rule mining
    df = pd.DataFrame(past_draws)
    df = df.stack().astype(str).str.get_dummies().groupby(level=0).sum()

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common = number_counts.most_common(total_numbers)
        predicted_main_numbers = [num for num, count in most_common]
    else:
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        last_draw = past_draws[-1]
        predicted_candidates = []

        for num in last_draw:
            relevant_rules = rules[rules['antecedents'].apply(lambda x: num in x)]
            if not relevant_rules.empty:
                top_rule = relevant_rules.sort_values(by='confidence', ascending=False).iloc[0]
                next_num = list(top_rule['consequents'])[0]
                predicted_candidates.append(next_num)

        predicted_main_numbers = list(dict.fromkeys(predicted_candidates))

        if len(predicted_main_numbers) < total_numbers:
            all_numbers = [num for draw in past_draws for num in draw]
            number_counts = Counter(all_numbers)
            most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_main_numbers]

            for num in most_common_numbers:
                predicted_main_numbers.append(num)
                if len(predicted_main_numbers) == total_numbers:
                    break

    predicted_main_numbers = [int(num) for num in predicted_main_numbers if min_num <= num <= max_num]

    if len(predicted_main_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_main_numbers:
                predicted_main_numbers.append(num)
                if len(predicted_main_numbers) == total_numbers:
                    break

    # Generate additional numbers if required
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_min = lottery_type_instance.additional_min_number
        additional_max = lottery_type_instance.additional_max_number
        additional_total = lottery_type_instance.additional_numbers_count

        all_additional_numbers = [num for draw in past_draws for num in draw if additional_min <= num <= additional_max]
        additional_counts = Counter(all_additional_numbers)
        additional_candidates = [num for num, count in additional_counts.most_common(additional_total)]

        if len(additional_candidates) < additional_total:
            for num in range(additional_min, additional_max + 1):
                if num not in additional_candidates:
                    additional_candidates.append(num)
                if len(additional_candidates) == additional_total:
                    break

        additional_numbers = additional_candidates[:additional_total]

    return sorted(predicted_main_numbers[:total_numbers]), sorted(additional_numbers)
