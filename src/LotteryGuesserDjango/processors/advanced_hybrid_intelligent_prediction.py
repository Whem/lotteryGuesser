# advanced_hybrid_intelligent_prediction.py
"""
Fejlett Hibrid Intelligens Lottószám Prediktor
Kombinálja a legjobb algoritmusokat intelligens súlyozással és adaptív tanulással
"""

import numpy as np
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from statistics import mean, median, stdev
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import logging

# Logging beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fejlett hibrid intelligens predikció fő és kiegészítő számokhoz.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_intelligent_hybrid_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            is_main=True
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_intelligent_hybrid_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                is_main=False
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba az advanced_hybrid_intelligent_prediction-ben: {e}")
        return generate_fallback_numbers(lottery_type_instance)


def generate_intelligent_hybrid_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    is_main: bool
) -> List[int]:
    """
    Intelligens hibrid számgenerálás kombinált módszerekkel.
    """
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, is_main)
    
    if len(past_draws) < 5:
        return generate_smart_random(min_number, max_number, pieces_of_draw_numbers)
    
    # Különböző predikciós módszerek alkalmazása
    prediction_methods = {
        'frequency_analysis': analyze_frequency_patterns,
        'statistical_trends': analyze_statistical_trends,
        'pattern_matching': analyze_pattern_sequences,
        'mathematical_sequences': analyze_mathematical_sequences,
        'gap_analysis': analyze_gap_patterns,
        'position_analysis': analyze_position_patterns,
        'sum_analysis': analyze_sum_patterns,
        'parity_analysis': analyze_parity_patterns
    }
    
    # Minden módszert futtatunk és súlyozunk
    method_predictions = {}
    method_weights = {}
    
    for method_name, method_func in prediction_methods.items():
        try:
            predictions, confidence = method_func(past_draws, min_number, max_number, pieces_of_draw_numbers)
            method_predictions[method_name] = predictions
            method_weights[method_name] = confidence
        except Exception as e:
            logger.warning(f"Hiba a {method_name} módszerben: {e}")
            method_predictions[method_name] = []
            method_weights[method_name] = 0.0
    
    # Súlyozott kombináció
    final_numbers = combine_predictions_intelligently(
        method_predictions, 
        method_weights, 
        min_number, 
        max_number, 
        pieces_of_draw_numbers
    )
    
    return final_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Történeti adatok lekérése és tisztítása."""
    try:
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id')[:200]  # Legutóbbi 200 húzás
        
        past_draws = []
        for draw in queryset:
            try:
                if is_main:
                    numbers = draw.lottery_type_number
                else:
                    numbers = getattr(draw, 'additional_numbers', None)
                
                if isinstance(numbers, list) and len(numbers) > 0:
                    valid_numbers = [int(num) for num in numbers if isinstance(num, (int, float))]
                    if valid_numbers:
                        past_draws.append(valid_numbers)
            except (ValueError, TypeError, AttributeError):
                continue
        
        return past_draws
    
    except Exception as e:
        logger.error(f"Hiba a történeti adatok lekérésében: {e}")
        return []


def analyze_frequency_patterns(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Frekvencia alapú elemzés időbeli súlyozással."""
    frequency = Counter()
    decay_factor = 0.95
    
    for i, draw in enumerate(past_draws):
        weight = decay_factor ** i
        for number in draw:
            if min_num <= number <= max_num:
                frequency[number] += weight
    
    # Leggyakoribb számok kiválasztása
    most_common = frequency.most_common(count * 2)
    predictions = [num for num, _ in most_common[:count]]
    
    # Megbízhatóság számítása
    confidence = min(1.0, len(most_common) / (max_num - min_num + 1) * 2)
    
    return predictions, confidence


def analyze_statistical_trends(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Statisztikai trendek elemzése."""
    if len(past_draws) < 10:
        return [], 0.0
    
    trends = defaultdict(list)
    for draw in past_draws:
        trends['mean'].append(mean(draw))
        trends['median'].append(median(draw))
        if len(draw) > 1:
            trends['stdev'].append(stdev(draw))
    
    # Trend predikciók
    predicted_mean = predict_next_value(trends['mean'])
    predicted_median = predict_next_value(trends['median'])
    
    # Számok generálása a trendek alapján
    predictions = []
    center = (predicted_mean + predicted_median) / 2
    spread = mean(trends['stdev']) if trends['stdev'] else (max_num - min_num) / 4
    
    for _ in range(count):
        num = int(np.random.normal(center, spread))
        num = max(min_num, min(num, max_num))
        if num not in predictions:
            predictions.append(num)
    
    confidence = min(1.0, len(past_draws) / 50)
    return predictions[:count], confidence


def analyze_pattern_sequences(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Szekvenciális minták elemzése."""
    sequence_patterns = Counter()
    
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for i in range(len(sorted_draw) - 1):
            diff = sorted_draw[i + 1] - sorted_draw[i]
            sequence_patterns[diff] += 1
    
    # Leggyakoribb különbségek
    common_diffs = [diff for diff, _ in sequence_patterns.most_common(5)]
    
    # Predikció generálása
    predictions = []
    if past_draws:
        last_numbers = sorted(past_draws[0])
        
        for base_num in last_numbers:
            for diff in common_diffs:
                new_num = base_num + diff
                if min_num <= new_num <= max_num and new_num not in predictions:
                    predictions.append(new_num)
                    if len(predictions) >= count:
                        break
            if len(predictions) >= count:
                break
    
    confidence = min(1.0, len(sequence_patterns) / 20)
    return predictions[:count], confidence


def analyze_mathematical_sequences(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Matematikai sorok elemzése (Fibonacci, prímek, stb.)."""
    # Fibonacci számok generálása
    fibonacci = generate_fibonacci_in_range(min_num, max_num)
    
    # Prím számok generálása
    primes = generate_primes_in_range(min_num, max_num)
    
    # Négyzetszámok
    squares = [i*i for i in range(1, int(max_num**0.5) + 1) if min_num <= i*i <= max_num]
    
    # Történeti adatokban előforduló matematikai számok
    math_numbers = set(fibonacci + primes + squares)
    
    # Frekvencia a matematikai számokban
    math_frequency = Counter()
    for draw in past_draws:
        for number in draw:
            if number in math_numbers:
                math_frequency[number] += 1
    
    predictions = [num for num, _ in math_frequency.most_common(count)]
    
    # Kiegészítés ha szükséges
    if len(predictions) < count:
        remaining_math = [num for num in math_numbers if num not in predictions]
        predictions.extend(random.sample(remaining_math, min(len(remaining_math), count - len(predictions))))
    
    confidence = min(1.0, len(math_frequency) / count * 0.5)
    return predictions[:count], confidence


def analyze_gap_patterns(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Hiányok és kihagyások elemzése."""
    recent_draws = past_draws[:20]  # Legutóbbi 20 húzás
    all_recent_numbers = set()
    for draw in recent_draws:
        all_recent_numbers.update(draw)
    
    # Hiányzó számok
    missing_numbers = [num for num in range(min_num, max_num + 1) if num not in all_recent_numbers]
    
    # Ritkán előforduló számok
    number_frequency = Counter()
    for draw in past_draws:
        for number in draw:
            number_frequency[number] += 1
    
    rare_numbers = [num for num, freq in number_frequency.items() 
                   if freq < len(past_draws) * 0.1 and min_num <= num <= max_num]
    
    # Kombináció
    gap_candidates = list(set(missing_numbers + rare_numbers))
    predictions = random.sample(gap_candidates, min(len(gap_candidates), count))
    
    confidence = min(1.0, len(gap_candidates) / count * 0.7)
    return predictions, confidence


def analyze_position_patterns(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Pozíció alapú minták elemzése."""
    position_frequency = defaultdict(Counter)
    
    for draw in past_draws:
        sorted_draw = sorted(draw)
        for pos, number in enumerate(sorted_draw):
            position_frequency[pos][number] += 1
    
    predictions = []
    for pos in range(min(count, len(position_frequency))):
        if pos in position_frequency:
            most_common_at_pos = position_frequency[pos].most_common(1)
            if most_common_at_pos:
                predictions.append(most_common_at_pos[0][0])
    
    confidence = min(1.0, len(predictions) / count)
    return predictions, confidence


def analyze_sum_patterns(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Összeg alapú minták elemzése."""
    draw_sums = [sum(draw) for draw in past_draws if len(draw) == count]
    
    if not draw_sums:
        return [], 0.0
    
    target_sum = predict_next_value(draw_sums)
    
    # Számok generálása a cél összeg körül
    predictions = []
    attempts = 0
    
    while len(predictions) < count and attempts < 1000:
        candidate_numbers = random.sample(range(min_num, max_num + 1), count)
        if abs(sum(candidate_numbers) - target_sum) < target_sum * 0.1:
            predictions = candidate_numbers
            break
        attempts += 1
    
    if not predictions:
        predictions = random.sample(range(min_num, max_num + 1), count)
    
    confidence = min(1.0, len(draw_sums) / 50)
    return predictions, confidence


def analyze_parity_patterns(past_draws: List[List[int]], min_num: int, max_num: int, count: int) -> Tuple[List[int], float]:
    """Páros/páratlan minták elemzése."""
    parity_patterns = Counter()
    
    for draw in past_draws:
        even_count = sum(1 for num in draw if num % 2 == 0)
        odd_count = len(draw) - even_count
        parity_patterns[(even_count, odd_count)] += 1
    
    # Leggyakoribb páros/páratlan arány
    most_common_parity = parity_patterns.most_common(1)
    if not most_common_parity:
        return [], 0.0
    
    target_even, target_odd = most_common_parity[0][0]
    
    # Számok generálása az arány alapján
    even_numbers = [num for num in range(min_num, max_num + 1) if num % 2 == 0]
    odd_numbers = [num for num in range(min_num, max_num + 1) if num % 2 == 1]
    
    predictions = []
    if len(even_numbers) >= target_even and len(odd_numbers) >= target_odd:
        predictions.extend(random.sample(even_numbers, min(target_even, len(even_numbers))))
        predictions.extend(random.sample(odd_numbers, min(target_odd, len(odd_numbers))))
    
    # Kiegészítés ha szükséges
    while len(predictions) < count:
        remaining = [num for num in range(min_num, max_num + 1) if num not in predictions]
        if remaining:
            predictions.append(random.choice(remaining))
        else:
            break
    
    confidence = min(1.0, len(parity_patterns) / 10)
    return predictions[:count], confidence


def combine_predictions_intelligently(
    method_predictions: Dict[str, List[int]], 
    method_weights: Dict[str, float],
    min_number: int,
    max_number: int,
    count: int
) -> List[int]:
    """Intelligens kombinálás súlyozással."""
    # Számok pontszámának kalkulálása
    number_scores = defaultdict(float)
    
    for method_name, predictions in method_predictions.items():
        weight = method_weights.get(method_name, 0.0)
        for i, number in enumerate(predictions):
            # Magasabb pozíció = magasabb pont
            position_bonus = (len(predictions) - i) / len(predictions) if predictions else 0
            number_scores[number] += weight * position_bonus
    
    # Legjobb számok kiválasztása
    sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    final_numbers = [num for num, score in sorted_numbers if min_number <= num <= max_number]
    
    # Diverzitás biztosítása
    final_numbers = ensure_diversity(final_numbers, min_number, max_number)
    
    # Kiegészítés ha szükséges
    while len(final_numbers) < count:
        remaining = [num for num in range(min_number, max_number + 1) if num not in final_numbers]
        if remaining:
            final_numbers.append(random.choice(remaining))
        else:
            break
    
    return final_numbers[:count]


def ensure_diversity(numbers: List[int], min_number: int, max_number: int) -> List[int]:
    """Diverzitás biztosítása a számok között."""
    if len(numbers) < 3:
        return numbers
    
    diverse_numbers = [numbers[0]]
    
    for num in numbers[1:]:
        # Ellenőrizzük, hogy legalább 2 távolságra van
        if all(abs(num - existing) >= 2 for existing in diverse_numbers):
            diverse_numbers.append(num)
    
    return diverse_numbers


def predict_next_value(values: List[float]) -> float:
    """Következő érték predikciója trend alapján."""
    if len(values) < 2:
        return values[0] if values else 0
    
    # Egyszerű lineáris trend
    if len(values) >= 3:
        recent_trend = (values[0] - values[2]) / 2  # Últimos 3 valores
        return values[0] + recent_trend
    else:
        return (values[0] + values[1]) / 2


def generate_fibonacci_in_range(min_num: int, max_num: int) -> List[int]:
    """Fibonacci számok generálása adott tartományban."""
    fibonacci = [1, 1]
    while fibonacci[-1] < max_num:
        next_fib = fibonacci[-1] + fibonacci[-2]
        if next_fib > max_num:
            break
        fibonacci.append(next_fib)
    
    return [num for num in fibonacci if min_num <= num <= max_num]


def generate_primes_in_range(min_num: int, max_num: int) -> List[int]:
    """Prím számok generálása adott tartományban."""
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    return [num for num in range(min_num, max_num + 1) if is_prime(num)]


def generate_smart_random(min_number: int, max_number: int, count: int) -> List[int]:
    """Intelligens véletlen számgenerálás normál eloszlással."""
    center = (min_number + max_number) / 2
    std = (max_number - min_number) / 6
    
    numbers = set()
    attempts = 0
    
    while len(numbers) < count and attempts < 1000:
        num = int(np.random.normal(center, std))
        if min_number <= num <= max_number:
            numbers.add(num)
        attempts += 1
    
    # Kiegészítés egyenletes eloszlással
    if len(numbers) < count:
        remaining = [num for num in range(min_number, max_number + 1) if num not in numbers]
        random.shuffle(remaining)
        numbers.update(remaining[:count - len(numbers)])
    
    return sorted(list(numbers)[:count])


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Fallback számgenerálás hiba esetén."""
    main_numbers = generate_smart_random(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_smart_random(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers) 