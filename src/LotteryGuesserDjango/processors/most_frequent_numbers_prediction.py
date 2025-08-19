# most_frequent_numbers_prediction.py
"""
Javított leggyakoribb számok predikció
Tartalmazz időbeli súlyozást, validációt és statisztikai elemzést
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
import logging
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

# Logging beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Javított leggyakoribb számok predikció időbeli súlyozással és validációval.
    Returns a tuple (main_numbers, additional_numbers).
    """
    try:
        # Fő számok generálása
        main_numbers = generate_improved_frequent_numbers(
            lottery_type_instance,
            min_number=int(lottery_type_instance.min_number),
            max_number=int(lottery_type_instance.max_number),
            pieces_of_draw_numbers=int(lottery_type_instance.pieces_of_draw_numbers),
            numbers_field='lottery_type_number'
        )

        # Kiegészítő számok generálása
        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = generate_improved_frequent_numbers(
                lottery_type_instance,
                min_number=int(lottery_type_instance.additional_min_number),
                max_number=int(lottery_type_instance.additional_max_number),
                pieces_of_draw_numbers=int(lottery_type_instance.additional_numbers_count),
                numbers_field='additional_numbers'
            )

        return sorted(main_numbers), sorted(additional_numbers)
    
    except Exception as e:
        logger.error(f"Hiba a most_frequent_numbers_prediction-ben: {e}")
        return generate_fallback_numbers(lottery_type_instance)

def generate_improved_frequent_numbers(
    lottery_type_instance: lg_lottery_type,
    min_number: int,
    max_number: int,
    pieces_of_draw_numbers: int,
    numbers_field: str
) -> List[int]:
    """
    Javított leggyakoribb számok generálása időbeli súlyozással és validációval.
    """
    # Történeti adatok lekérése
    past_draws = get_historical_data(lottery_type_instance, numbers_field)
    
    if len(past_draws) < 10:
        logger.warning(f"Kevés történeti adat ({len(past_draws)} húzás), intelligens véletlen generálás")
        return generate_intelligent_random(min_number, max_number, pieces_of_draw_numbers)
    
    # Időbeli súlyozással számított frekvencia
    weighted_frequency = calculate_weighted_frequency(past_draws, min_number, max_number)
    
    # Statisztikai validáció
    validated_frequency = apply_statistical_validation(weighted_frequency, past_draws)
    
    # Számok kiválasztása
    selected_numbers = select_numbers_with_diversity(
        validated_frequency, min_number, max_number, pieces_of_draw_numbers
    )
    
    # Végső validáció
    final_numbers = validate_final_selection(
        selected_numbers, min_number, max_number, pieces_of_draw_numbers
    )
    
    return final_numbers


def get_historical_data(lottery_type_instance: lg_lottery_type, numbers_field: str) -> List[List[int]]:
    """
    Történeti adatok lekérése és tisztítása determinisztikus feldolgozással.
    """
    try:
        # Legutóbbi 150 húzás lekérése
        queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list(numbers_field, flat=True)[:150]
        
        all_past_draws = []
        for draw in queryset:
            if isinstance(draw, list) and len(draw) > 0:
                # Csak érvényes számokat tartunk meg
                valid_numbers = []
                for num in draw:
                    try:
                        int_num = int(num)
                        if isinstance(int_num, int):
                            valid_numbers.append(int_num)
                    except (ValueError, TypeError):
                        continue
                
                if valid_numbers:
                    all_past_draws.append(valid_numbers)
        
        # Determinisztikus mintavételezés: a legutóbbi ~70% húzás megtartása
        if len(all_past_draws) > 20:
            sample_size = max(20, int(len(all_past_draws) * 0.7))
            past_draws = all_past_draws[:sample_size]
        else:
            past_draws = all_past_draws
        
        return past_draws
    
    except Exception as e:
        logger.error(f"Hiba a történeti adatok lekérésében: {e}")
        return []


def calculate_weighted_frequency(past_draws: List[List[int]], min_number: int, max_number: int) -> Counter:
    """
    Időbeli súlyozással számított frekvencia.
    Újabb húzások nagyobb súlyt kapnak.
    """
    weighted_counter = Counter()
    decay_factor = 0.95  # Időbeli súlyozás faktora
    
    for i, draw in enumerate(past_draws):
        # Időbeli súly (újabb húzások nagyobb súlyt kapnak)
        weight = decay_factor ** i
        
        for number in draw:
            if min_number <= number <= max_number:
                weighted_counter[number] += weight
    
    return weighted_counter


def apply_statistical_validation(frequency: Counter, past_draws: List[List[int]]) -> Counter:
    """
    Statisztikai validáció alkalmazása.
    Outlier számok súlyának csökkentése.
    """
    if not past_draws:
        return frequency
    
    # Átlag és szórás számítása
    all_numbers = [num for draw in past_draws for num in draw]
    if not all_numbers:
        return frequency
    
    mean_val = np.mean(all_numbers)
    std_val = np.std(all_numbers)
    
    validated_frequency = Counter()
    
    for number, count in frequency.items():
        # Z-score számítása
        z_score = abs(number - mean_val) / max(std_val, 1)
        
        # Outlier számok súlyának csökkentése
        if z_score > 2.0:
            validated_frequency[number] = count * 0.7  # 30% csökkentés
        else:
            validated_frequency[number] = count
    
    return validated_frequency


def select_numbers_with_diversity(frequency: Counter, min_number: int, max_number: int, 
                                pieces_of_draw_numbers: int) -> List[int]:
    """
    Számok kiválasztása diverzitással determinisztikusan.
    """
    # Leggyakoribb számok (bővebb választás)
    top_candidates_count = min(pieces_of_draw_numbers * 3, len(frequency))
    top_candidates = [num for num, _ in frequency.most_common(top_candidates_count)]
    
    # Determinisztikus kiválasztás a top jelöltek közül (70% top + 30% determinisztikus kiegészítés)
    guaranteed_count = max(1, int(pieces_of_draw_numbers * 0.7))
    selected = top_candidates[:guaranteed_count]
    
    # Kiegészítés determinisztikusan a maradék helyekre
    remaining_needed = pieces_of_draw_numbers - len(selected)
    if remaining_needed > 0:
        remaining_candidates = top_candidates[guaranteed_count:]
        if len(remaining_candidates) >= remaining_needed:
            selected.extend(remaining_candidates[:remaining_needed])
        else:
            selected.extend(remaining_candidates)
            # Ha még mindig kell több, determinisztikusan a legkisebb elérhető számokkal töltjük fel
            all_numbers = set(range(min_number, max_number + 1))
            unused_numbers = sorted(list(all_numbers - set(selected)))
            if unused_numbers and len(selected) < pieces_of_draw_numbers:
                needed = pieces_of_draw_numbers - len(selected)
                selected.extend(unused_numbers[:needed])
    
    # Diverzitás biztosítása
    if len(selected) >= 3:
        selected = ensure_number_diversity(selected, min_number, max_number)
    
    return selected[:pieces_of_draw_numbers]


def ensure_number_diversity(numbers: List[int], min_number: int, max_number: int) -> List[int]:
    """
    Számok diverzitásának biztosítása.
    """
    if len(numbers) < 3:
        return numbers
    
    sorted_numbers = sorted(numbers)
    diverse_numbers = [sorted_numbers[0]]  # Első szám mindig benne van
    
    for num in sorted_numbers[1:]:
        # Ellenőrizzük, hogy legalább 2 távolságra van az utolsó kiválasztott számtól
        if abs(num - diverse_numbers[-1]) >= 2:
            diverse_numbers.append(num)
    
    # Ha túl kevés diverzitást értünk el, kiegészítjük
    if len(diverse_numbers) < len(numbers):
        remaining = [num for num in range(min_number, max_number + 1)
                     if num not in diverse_numbers]
        # Determinisztikus kiegészítés: a legkisebb elérhető számokkal
        diverse_numbers.extend(remaining[:len(numbers) - len(diverse_numbers)])
    
    return diverse_numbers[:len(numbers)]


def validate_final_selection(selected_numbers: List[int], min_number: int, max_number: int, 
                           pieces_of_draw_numbers: int) -> List[int]:
    """
    Végső validáció és kiegészítés.
    """
    # Duplikátumok eltávolítása
    unique_numbers = list(set(selected_numbers))
    
    # Tartomány ellenőrzés
    valid_numbers = [num for num in unique_numbers if min_number <= num <= max_number]
    
    # Kiegészítés szükség esetén
    if len(valid_numbers) < pieces_of_draw_numbers:
        remaining_numbers = [
            num for num in range(min_number, max_number + 1) 
            if num not in valid_numbers
        ]
        # Determinisztikus kiegészítés: a legkisebb elérhető számokkal
        valid_numbers.extend(remaining_numbers[:pieces_of_draw_numbers - len(valid_numbers)])
    
    # Típus konverzió
    final_numbers = [int(num) for num in valid_numbers[:pieces_of_draw_numbers]]
    
    return final_numbers


def generate_intelligent_random(min_number: int, max_number: int, count: int) -> List[int]:
    """
    Determinisztikus "intelligens" generálás: középről kifelé terjedő mintázattal.
    """
    if count <= 0 or min_number > max_number:
        return []
    center = int(round((min_number + max_number) / 2))
    selected: List[int] = []
    used = set()
    # Offsetek: 0, +1, -1, +2, -2, ...
    k = 0
    while len(selected) < count:
        if k == 0:
            offsets = [0]
        else:
            offsets = [k, -k]
        for off in offsets:
            candidate = center + off
            if min_number <= candidate <= max_number and candidate not in used:
                selected.append(int(candidate))
                used.add(int(candidate))
                if len(selected) >= count:
                    break
        k += 1
        # Biztonsági megszakítás szélsőséges esetben
        if center - k < min_number and center + k > max_number and len(selected) < count:
            # Töltés a legkisebb elérhetőkkel
            for c in range(min_number, max_number + 1):
                if c not in used:
                    selected.append(int(c))
                    used.add(int(c))
                    if len(selected) >= count:
                        break
    return selected[:count]


def generate_fallback_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Fallback számgenerálás hiba esetén.
    """
    main_numbers = generate_intelligent_random(
        int(lottery_type_instance.min_number),
        int(lottery_type_instance.max_number),
        int(lottery_type_instance.pieces_of_draw_numbers)
    )
    
    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_intelligent_random(
            int(lottery_type_instance.additional_min_number),
            int(lottery_type_instance.additional_max_number),
            int(lottery_type_instance.additional_numbers_count)
        )
    
    return sorted(main_numbers), sorted(additional_numbers)
