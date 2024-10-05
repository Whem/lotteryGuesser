# recurrence_quantification_analysis_prediction.py

import numpy as np
from pyunicorn.timeseries import RecurrencePlot
from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Rekurrens Kvantifikációs Analízis (RQA) alkalmazásával.

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

    # Rekurrens Kvantifikációs Analízis végrehajtása
    # Rekurrencia plot létrehozása
    rp = RecurrencePlot(
        time_series,
        dim=1,
        tau=1,
        metric='euclidean',
        normalize=False,
        recurrence_rate=0.1  # Adjunk meg egy rekurrencia arányt
    )

    # Rekurrencia mátrix lekérése
    recurrence_matrix = rp.recurrence_matrix()

    # Számoljuk ki a rekurrencia gyakoriságát minden időpontra
    recurrence_histogram = np.sum(recurrence_matrix, axis=0)

    # Társítjuk a gyakorisági értékeket a számokhoz
    number_scores = {}
    for idx, count in enumerate(recurrence_histogram):
        number = int(time_series[idx])
        if min_num <= number <= max_num:
            number_scores[number] = number_scores.get(number, 0) + count

    # Rendezzük a számokat a rekurrencia gyakoriságuk alapján
    sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)

    # Kiválasztjuk az első 'total_numbers' számot
    predicted_numbers = [num for num, score in sorted_numbers]

    # Eltávolítjuk a duplikátumokat és csak az érvényes számokat tartjuk meg
    predicted_numbers = [
        int(num) for num in dict.fromkeys(predicted_numbers)
        if min_num <= num <= max_num
    ]

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = time_series
        number_counts = {}
        for number in all_numbers:
            number = int(number)
            if min_num <= number <= max_num:
                number_counts[number] = number_counts.get(number, 0) + 1
        sorted_counts = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_counts:
            num = int(num)
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
