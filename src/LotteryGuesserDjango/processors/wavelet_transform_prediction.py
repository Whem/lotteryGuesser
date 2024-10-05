# wavelet_transform_prediction.py

import numpy as np
import pywt  # Győződj meg róla, hogy a PyWavelets könyvtár telepítve van
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat a múltbeli húzások hullámtranszformációs elemzése alapján.

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

    if len(past_draws) < 10:
        # Ha nincs elég adat, visszaadjuk a legkisebb 'total_numbers' számot
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Átalakítjuk a múltbeli húzásokat egy mátrixba
    draw_matrix = np.array(past_draws)

    # Minden pozícióra külön sorozatot készítünk
    position_series = [draw_matrix[:, i] for i in range(total_numbers)]

    # A hullámtranszformáció alkalmazása minden pozícióra
    predicted_numbers = []
    for series in position_series:
        # Diszkrét hullámtranszformáció
        coeffs = pywt.wavedec(series, 'db1', level=2)

        # A részletező együtthatók elhanyagolása (alacsony frekvenciás komponensek megtartása)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]

        # Jel rekonstruálása
        reconstructed_series = pywt.waverec(coeffs, 'db1')

        # Az utolsó érték előrejelzése lineáris trend alapján
        if len(reconstructed_series) >= 2:
            trend = reconstructed_series[-1] - reconstructed_series[-2]
            next_value = reconstructed_series[-1] + trend
        else:
            next_value = reconstructed_series[-1]

        # A következő érték kerekítése és a tartományhoz igazítása
        predicted_number = int(round(next_value))
        if predicted_number < min_num:
            predicted_number = min_num
        elif predicted_number > max_num:
            predicted_number = max_num

        predicted_numbers.append(predicted_number)

    # Egyedivé tesszük a számokat és rendezzük őket
    predicted_numbers = list(set(predicted_numbers))
    predicted_numbers.sort()

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

    # Végső rendezés és levágás a szükséges számú elemre
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
