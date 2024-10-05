# deterministic_trend_pattern_predictor.py

from collections import defaultdict, Counter
from django.apps import apps
import numpy as np

def analyze_frequency(past_draws, min_num, max_num):
    frequency = Counter(num for draw in past_draws for num in draw)
    return [frequency.get(num, 0) for num in range(min_num, max_num + 1)]

def analyze_recency(past_draws, min_num, max_num):
    last_seen = defaultdict(lambda: float('inf'))
    for i, draw in enumerate(reversed(past_draws)):
        for num in draw:
            if last_seen[num] == float('inf'):
                last_seen[num] = i
    return [last_seen.get(num, float('inf')) for num in range(min_num, max_num + 1)]

def analyze_patterns(past_draws):
    patterns = defaultdict(int)
    for i in range(len(past_draws) - 1):
        for num in past_draws[i]:
            if num in past_draws[i+1]:
                patterns[num] += 1
    return patterns

def analyze_trends(past_draws, min_num, max_num):
    trends = [[] for _ in range(min_num, max_num + 1)]
    for draw in past_draws:
        for num in range(min_num, max_num + 1):
            trends[num - min_num].append(1 if num in draw else 0)
    return [sum(trend[-5:]) for trend in trends]  # Consider last 5 draws for trend

def get_numbers(lottery_type_instance):
    lg_lottery_winner_number = apps.get_model('algorithms', 'lg_lottery_winner_number')

    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:100].values_list('lottery_type_number', flat=True))

    if len(past_draws) < 10:
        # Not enough data for analysis, return the most common numbers
        all_numbers = [num for draw in past_draws for num in draw]
        return sorted(Counter(all_numbers).most_common(total_numbers))

    frequency = analyze_frequency(past_draws, min_num, max_num)
    recency = analyze_recency(past_draws, min_num, max_num)
    patterns = analyze_patterns(past_draws)
    trends = analyze_trends(past_draws, min_num, max_num)

    # Normalize and combine scores
    freq_score = np.array(frequency) / max(frequency)
    rec_score = 1 - (np.array(recency) / max(recency))  # Invert so recent numbers have higher score
    pat_score = np.array([patterns.get(num, 0) for num in range(min_num, max_num + 1)]) / max(patterns.values() or [1])
    trend_score = np.array(trends) / 5  # 5 is the maximum trend score

    # Weighted combination of scores
    final_score = (0.3 * freq_score + 0.2 * rec_score + 0.2 * pat_score + 0.3 * trend_score)

    # Select top scoring numbers
    top_numbers = sorted(range(len(final_score)), key=lambda i: final_score[i], reverse=True)[:total_numbers]

    return sorted(num + min_num for num in top_numbers)