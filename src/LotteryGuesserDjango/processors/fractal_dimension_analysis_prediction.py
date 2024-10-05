# fractal_dimension_analysis_prediction.py

import numpy as np
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Fraktáldimenzió-elemzés alkalmazásával.

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

    # Átalakítjuk a múltbeli húzásokat egy idősorozattá
    draw_matrix = np.array(past_draws)
    time_series = draw_matrix.flatten()

    # Fraktáldimenzió kiszámítása a Higuchi-módszerrel
    max_k = 10  # Maximális időintervallum
    N = len(time_series)
    Lk = np.zeros(max_k)
    for k in range(1, max_k+1):
        Lm = []
        for m in range(k):
            Lmk = 0
            n_max = int(np.floor((N - m) / k))
            if n_max < 2:
                continue
            for i in range(1, n_max):
                Lmk += abs(time_series[m + i*k] - time_series[m + (i-1)*k])
            norm = (N - 1) / (k * n_max * k)
            Lmk = (Lmk * norm)
            Lm.append(Lmk)
        if Lm:
            Lk[k-1] = np.sum(Lm) / len(Lm)
        else:
            Lk[k-1] = 0

    # Lineáris regresszió a fraktáldimenzió becsléséhez
    positive_indices = Lk > 0
    ln_Lk = np.log(Lk[positive_indices])
    ln_k = np.log(np.arange(1, max_k+1)[positive_indices])
    if len(ln_k) >= 2:
        coeffs = np.polyfit(ln_k, ln_Lk, 1)
        fractal_dimension = coeffs[0]
    else:
        fractal_dimension = 1  # Alapértelmezett érték, ha nincs elég adat

    # Számok gyakoriságának módosítása a fraktáldimenzió alapján
    number_counts = {}
    for number in time_series:
        number = int(number)
        if min_num <= number <= max_num:
            number_counts[number] = number_counts.get(number, 0) + 1

    adjusted_counts = {}
    for number, count in number_counts.items():
        adjusted_counts[number] = count * fractal_dimension

    # Számok rendezése a módosított gyakoriság alapján
    sorted_numbers = sorted(adjusted_counts.items(), key=lambda x: x[1], reverse=True)

    # Kiválasztjuk az első 'total_numbers' számot
    predicted_numbers = [int(num) for num, score in sorted_numbers[:total_numbers]]

    # Eltávolítjuk a duplikátumokat és csak az érvényes számokat tartjuk meg
    predicted_numbers = [
        int(num) for num in dict.fromkeys(predicted_numbers)
        if min_num <= num <= max_num
    ]

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a következő leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        for num, _ in sorted_numbers:
            num = int(num)
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    predicted_numbers.sort()
    return predicted_numbers
