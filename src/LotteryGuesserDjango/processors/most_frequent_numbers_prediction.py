# most_frequent_numbers_prediction.py

from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat a múltbeli húzások leggyakrabban előforduló számai alapján.

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

    if not past_draws:
        # Ha nincs elég adat, visszaadjuk a legkisebb 'total_numbers' számot
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Laposítjuk a listát, hogy minden számot megkapjunk
    all_numbers = [number for draw in past_draws for number in draw]

    # Számoljuk az egyes számok előfordulási gyakoriságát
    number_counts = Counter(all_numbers)

    # Kiválasztjuk a leggyakoribb számokat
    most_common_numbers = [num for num, count in number_counts.most_common()]

    # Kiválasztjuk az első 'total_numbers' számot
    selected_numbers = most_common_numbers[:total_numbers]

    # Ha nincs elég szám, kiegészítjük a hiányzó számokkal
    if len(selected_numbers) < total_numbers:
        remaining_numbers = [
            num for num in range(min_num, max_num + 1) if num not in selected_numbers
        ]
        selected_numbers.extend(remaining_numbers[:total_numbers - len(selected_numbers)])

    selected_numbers.sort()
    return selected_numbers
