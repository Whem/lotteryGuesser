# empirical_mode_decomposition_prediction.py

import numpy as np
from PyEMD import EMD
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Empirikus Módus Dekompozíció (EMD) segítségével.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.

    Visszatérési érték:
    - Egy rendezett lista a megjósolt lottószámokról.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Lekérjük a múltbeli nyerőszámokat
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = [
        draw for draw in past_draws_queryset
        if isinstance(draw, list) and len(draw) == total_numbers
    ]

    if len(past_draws) < 20:
        # Ha nincs elég adat, visszaadjuk a legkisebb 'total_numbers' számot
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Átalakítjuk a múltbeli húzásokat egy numpy mátrixba
    draw_matrix = np.array(past_draws)

    predicted_numbers = []

    # Minden pozícióra külön EMD alkalmazása
    for i in range(total_numbers):
        # Kiválasztjuk az adott pozíció idősorozatát
        series = draw_matrix[:, i]

        # EMD alkalmazása
        emd = EMD()
        IMFs = emd.emd(series)

        # Rekonstruáljuk a jelet az első IMF nélkül (zaj szűrése)
        if IMFs.shape[0] > 1:
            reconstructed_series = np.sum(IMFs[1:], axis=0)
        else:
            reconstructed_series = series

        # Lineáris trend alapján előrejelzés
        if len(reconstructed_series) >= 2:
            trend = reconstructed_series[-1] - reconstructed_series[-2]
            next_value = reconstructed_series[-1] + trend
        else:
            next_value = reconstructed_series[-1]

        # Kerekítés és tartományhoz igazítás
        predicted_number = int(round(next_value))
        if predicted_number < min_num:
            predicted_number = min_num
        elif predicted_number > max_num:
            predicted_number = max_num

        predicted_numbers.append(predicted_number)

    # Egyedivé tesszük a számokat
    predicted_numbers = list(set(predicted_numbers))

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_draws for number in draw]
        number_counts = {}
        for number in all_numbers:
            number_counts[number] = number_counts.get(number, 0) + 1
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_numbers:
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Rendezés és levágás a szükséges hosszra
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers

