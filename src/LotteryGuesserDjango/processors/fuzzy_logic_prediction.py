# fuzzy_logic_prediction.py

import numpy as np
from collections import defaultdict, Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Fuzzy Logika alapú elemzéssel.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.

    Visszatérési érték:
    - Egy rendezett lista a megjósolt lottószámokról.
    """
    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

    # Lekérjük a múltbeli nyerőszámokat
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [
        [int(num) for num in draw] for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        # Ha nincs elég adat, visszaadjuk a legkisebb 'total_numbers' számot
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Számok gyakoriságának számolása
    all_numbers = [number for draw in past_draws for number in draw]
    number_counts = Counter(all_numbers)
    most_common_numbers = [num for num, count in number_counts.most_common()]

    # Fuzzy Halmazok Definiálása
    def low_membership(x):
        if x <= min_num + (max_num - min_num) / 3:
            return 1.0
        elif x <= min_num + 2 * (max_num - min_num) / 3:
            return (min_num + 2 * (max_num - min_num) / 3 - x) / ((max_num - min_num) / 3)
        else:
            return 0.0

    def medium_membership(x):
        if x <= min_num + (max_num - min_num) / 3 or x >= min_num + 2 * (max_num - min_num) / 3:
            return 0.0
        else:
            return 1.0

    def high_membership(x):
        if x >= min_num + 2 * (max_num - min_num) / 3:
            return 1.0
        elif x >= min_num + (max_num - min_num) / 3:
            return (x - (min_num + (max_num - min_num) / 3)) / ((max_num - min_num) / 3)
        else:
            return 0.0

    # Fuzzy Szabályok Kialakítása
    # Példa szabályok:
    # 1. Ha egy szám LOW, akkor a következő szám valószínűleg LOW vagy MEDIUM.
    # 2. Ha egy szám MEDIUM, akkor a következő szám valószínűleg MEDIUM vagy HIGH.
    # 3. Ha egy szám HIGH, akkor a következő szám valószínűleg MEDIUM vagy HIGH.

    rules = {
        'LOW': ['LOW', 'MEDIUM'],
        'MEDIUM': ['MEDIUM', 'HIGH'],
        'HIGH': ['MEDIUM', 'HIGH']
    }

    # Utolsó húzás számainak lekérése
    last_draw = past_draws[-1]

    predicted_numbers = []

    for pos, last_num in enumerate(last_draw):
        # Meghatározzuk a last_num fuzzy kategóriáit
        memberships = {
            'LOW': low_membership(last_num),
            'MEDIUM': medium_membership(last_num),
            'HIGH': high_membership(last_num)
        }

        # Aggregáljuk a lehetséges következő számokat a szabályok alapján
        possible_next_numbers = defaultdict(float)

        for category, degree in memberships.items():
            if degree > 0:
                for consequent in rules[category]:
                    # Válasszuk a leggyakoribb számokat a következő kategóriába tartozóból
                    for num in most_common_numbers:
                        if consequent == 'LOW' and low_membership(num) > 0.5:
                            possible_next_numbers[num] += degree * number_counts[num]
                        elif consequent == 'MEDIUM' and medium_membership(num) > 0.5:
                            possible_next_numbers[num] += degree * number_counts[num]
                        elif consequent == 'HIGH' and high_membership(num) > 0.5:
                            possible_next_numbers[num] += degree * number_counts[num]

        if possible_next_numbers:
            # Válasszuk a legmagasabb értékkel rendelkező számot
            next_num = max(possible_next_numbers, key=possible_next_numbers.get)
            predicted_numbers.append(next_num)
        else:
            # Ha nem találunk megfelelő számot, válasszuk a leggyakoribb számot
            if most_common_numbers:
                predicted_numbers.append(most_common_numbers[0])
            else:
                predicted_numbers.append(min_num)

    # Eltávolítjuk a duplikátumokat
    predicted_numbers = list(dict.fromkeys(predicted_numbers))

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        for num in most_common_numbers:
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
