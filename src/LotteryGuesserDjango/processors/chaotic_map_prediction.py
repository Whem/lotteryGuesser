# chaotic_map_prediction.py

import numpy as np
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Deterministic Chaos-Based elemzéssel.

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

    # Kaotikus Térkép Konfigurálása (Logistic Map)
    r = 3.9  # Kaotikus térkép paramétere
    iterations = 1000  # Iterációk száma
    window_size = 100  # Az előrejelzéshez használt utolsó iterációk száma

    # Kezdő állapot beállítása az utolsó húzás számainak átlagával
    last_draw = past_draws[-1]
    x0 = np.mean(last_draw) / (max_num + 1)  # Normalizálás 0 és 1 közé

    # Iterációk futtatása
    x = x0
    sequence = []
    for _ in range(iterations):
        x = r * x * (1 - x)
        sequence.append(x)

    # Az utolsó 'window_size' iteráció átlagát használjuk
    recent_sequence = sequence[-window_size:]
    average_recent = np.mean(recent_sequence)

    # Mapping az előrejelzésekhez
    # Skálázzuk az átlagos értéket a megengedett szám tartományára
    predicted_num = int(round(average_recent * (max_num - min_num + 1))) + min_num

    # Ellenőrzés és korrekció
    if predicted_num < min_num:
        predicted_num = min_num
    elif predicted_num > max_num:
        predicted_num = max_num

    predicted_numbers = [predicted_num]

    # Duplikátumok és tartomány ellenőrzése
    predicted_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

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
