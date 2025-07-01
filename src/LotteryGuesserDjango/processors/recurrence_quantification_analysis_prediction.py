# recurrence_quantification_analysis_prediction.py
"""
Optimalizált Recurrence Quantification Analysis 
Saját implementáció pyunicorn helyett a gyorsaság és függőségmentesség érdekében
"""

import numpy as np
from typing import List, Tuple
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
from scipy.spatial.distance import pdist, squareform
from collections import Counter


def create_recurrence_matrix(time_series: np.ndarray, threshold: float = None, metric: str = 'euclidean') -> np.ndarray:
    """
    Egyszerű és gyors recurrence matrix létrehozás
    """
    n = len(time_series)
    
    # Embed the time series (phase space reconstruction)
    embedded = np.array([time_series[i:i+3] for i in range(n-2)])
    
    # Calculate distance matrix
    distances = squareform(pdist(embedded, metric=metric))
    
    # Determine threshold if not provided
    if threshold is None:
        threshold = np.percentile(distances, 10)  # 10th percentile as threshold
    
    # Create recurrence matrix
    recurrence_matrix = distances <= threshold
    
    return recurrence_matrix.astype(int)


def fast_recurrence_quantification(recurrence_matrix: np.ndarray) -> dict:
    """
    Gyors RQA mutatók számítása
    """
    n = recurrence_matrix.shape[0]
    
    # Recurrence rate
    recurrence_rate = np.sum(recurrence_matrix) / (n * n)
    
    # Determinism (percentage of recurrent points forming diagonal lines)
    diagonal_lengths = []
    for i in range(n):
        for j in range(n):
            if recurrence_matrix[i, j] == 1:
                length = 1
                k = 1
                while (i + k < n and j + k < n and recurrence_matrix[i + k, j + k] == 1):
                    length += 1
                    k += 1
                if length >= 2:  # Minimum line length
                    diagonal_lengths.append(length)
    
    determinism = len(diagonal_lengths) / max(1, np.sum(recurrence_matrix)) if diagonal_lengths else 0
    
    # Average diagonal line length
    avg_diagonal_length = np.mean(diagonal_lengths) if diagonal_lengths else 0
    
    # Laminarity (percentage of recurrent points forming vertical lines)
    vertical_lengths = []
    for j in range(n):
        i = 0
        while i < n:
            if recurrence_matrix[i, j] == 1:
                length = 1
                i += 1
                while i < n and recurrence_matrix[i, j] == 1:
                    length += 1
                    i += 1
                if length >= 2:
                    vertical_lengths.append(length)
            else:
                i += 1
    
    laminarity = len(vertical_lengths) / max(1, np.sum(recurrence_matrix)) if vertical_lengths else 0
    
    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diagonal_length': avg_diagonal_length,
        'laminarity': laminarity
    }


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Optimalizált RQA alapú számgenerálás
    """
    # Generate main numbers
    main_numbers = generate_number_set(
        lottery_type_instance=lottery_type_instance,
        number_field='lottery_type_number',
        min_num=int(lottery_type_instance.min_number),
        max_num=int(lottery_type_instance.max_number),
        total_numbers=int(lottery_type_instance.pieces_of_draw_numbers)
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        # Generate additional numbers
        additional_numbers = generate_number_set(
            lottery_type_instance=lottery_type_instance,
            number_field='additional_numbers',
            min_num=int(lottery_type_instance.additional_min_number),
            max_num=int(lottery_type_instance.additional_max_number),
            total_numbers=int(lottery_type_instance.additional_numbers_count)
        )

    return main_numbers, additional_numbers


def generate_number_set(
    lottery_type_instance: lg_lottery_type,
    number_field: str,
    min_num: int,
    max_num: int,
    total_numbers: int
) -> List[int]:
    """
    Gyorsított RQA alapú számgenerálás
    """
    # Kevesebb húzás a gyorsaság érdekében
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('-id')[:50].values_list(number_field, flat=True)  # Csak 50 húzás

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 10:  # Csökkentett minimum
        return list(range(min_num, min_num + total_numbers))

    # Gyorsabb idősor létrehozás
    time_series = []
    for draw in past_draws:
        time_series.extend(sorted(draw))
    
    time_series = np.array(time_series)
    
    if len(time_series) < 10:
        return list(range(min_num, min_num + total_numbers))

    try:
        # Gyors recurrence analysis
        recurrence_matrix = create_recurrence_matrix(time_series, threshold=None)
        rqa_measures = fast_recurrence_quantification(recurrence_matrix)
        
        # Számláló alapú scoring
        number_scores = Counter()
        
        # Használjuk az RQA mérőszámokat a súlyozáshoz
        weights = {
            'recurrence_rate': rqa_measures['recurrence_rate'],
            'determinism': rqa_measures['determinism'],
            'laminarity': rqa_measures['laminarity']
        }
        
        # Egyszerűsített scoring
        for i, num in enumerate(time_series):
            if min_num <= num <= max_num:
                # Recency weighting
                recency_weight = 1.0 / (i // total_numbers + 1)
                
                # RQA-based scoring
                rqa_score = (weights['recurrence_rate'] * 2 + 
                           weights['determinism'] + 
                           weights['laminarity'])
                
                final_score = recency_weight * rqa_score
                number_scores[int(num)] += final_score
        
        # Kiválasztás a legmagasabb pontszámok alapján
        if number_scores:
            sorted_numbers = [num for num, _ in number_scores.most_common()]
            predicted_numbers = []
            
            for num in sorted_numbers:
                if min_num <= num <= max_num and num not in predicted_numbers:
                    predicted_numbers.append(num)
                    if len(predicted_numbers) >= total_numbers:
                        break
            
            # Feltöltés ha szükséges
            if len(predicted_numbers) < total_numbers:
                all_numbers = list(range(min_num, max_num + 1))
                remaining = [x for x in all_numbers if x not in predicted_numbers]
                needed = total_numbers - len(predicted_numbers)
                predicted_numbers.extend(remaining[:needed])
            
            return sorted(predicted_numbers[:total_numbers])
        
    except Exception as e:
        print(f"RQA hiba: {e}")
    
    # Fallback: egyszerű frekvencia alapú
    all_numbers = [num for draw in past_draws for num in draw]
    number_counts = Counter(all_numbers)
    
    predicted_numbers = []
    for num, _ in number_counts.most_common():
        if min_num <= num <= max_num and num not in predicted_numbers:
            predicted_numbers.append(num)
            if len(predicted_numbers) >= total_numbers:
                break
    
    # Végső feltöltés
    while len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(num)
                break
    
    return sorted(predicted_numbers[:total_numbers])
