# fourier_transform_prediction.py

import numpy as np
from scipy.fft import fft, ifft
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Fourier-transzformáció alkalmazásával.

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

    # Átalakítjuk a múltbeli húzásokat egy numpy mátrixba
    draw_matrix = np.array(past_draws)

    predicted_numbers = []

    # Minden pozícióra külön Fourier-elemzést végzünk
    for i in range(total_numbers):
        # Kiválasztjuk az adott pozíció idősorozatát
        series = draw_matrix[:, i]

        # Fourier-transzformáció alkalmazása
        series_fft = fft(series)

        # Alacsony frekvenciájú komponensek megtartása
        n = len(series_fft)
        cutoff = n // 5  # Csak az első 20% frekvenciát tartjuk meg
        series_fft_filtered = np.zeros_like(series_fft)
        series_fft_filtered[:cutoff] = series_fft[:cutoff]

        # Inverz Fourier-transzformáció a jel rekonstrukciójához
        series_filtered = ifft(series_fft_filtered).real

        # Következő érték előrejelzése lineáris trend alapján
        if len(series_filtered) >= 2:
            trend = series_filtered[-1] - series_filtered[-2]
            next_value = series_filtered[-1] + trend
        else:
            next_value = series_filtered[-1]

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