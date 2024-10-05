# cellular_automaton_prediction.py

import numpy as np
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Cellular Automaton (CA) alapú elemzéssel.

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

    # Leggyakoribb számok számolása
    all_numbers = [number for draw in past_draws for number in draw]
    number_counts = Counter(all_numbers)
    most_common_numbers = [num for num, count in number_counts.most_common()]

    # Cellular Automaton Konfigurálása
    # Minden pozícióra külön CA-t alkalmazunk
    predicted_numbers = []
    for pos in range(total_numbers):
        # Kiválasztjuk az adott pozíció utolsó húzásának számát
        last_num = past_draws[-1][pos]

        # Átalakítjuk a múltbeli húzásokat egy bináris sorozattá
        # Minden szám jelenléte jelzi az állapotot
        state_length = max_num - min_num + 1
        state = np.zeros(state_length, dtype=int)
        for draw in past_draws:
            num = draw[pos]
            state[num - min_num] = 1  # Jelöljük a jelenlévést

        # CA Szabálykiválasztás (Wolfram szabály 30 például)
        rule_number = 30
        rule = np.binary_repr(rule_number, width=8)
        rule = np.array([int(x) for x in rule], dtype=int)

        def apply_rule(left, center, right, rule):
            """Alkalmazza a CA szabályát egy cellára."""
            index = (left << 2) | (center << 1) | right
            return rule[7 - index]

        # CA iterációk száma
        iterations = 10

        current_state = state.copy()
        for _ in range(iterations):
            next_state = current_state.copy()
            for i in range(1, state_length - 1):
                left = current_state[i - 1]
                center = current_state[i]
                right = current_state[i + 1]
                next_state[i] = apply_rule(left, center, right, rule)
            current_state = next_state

        # Az utolsó állapot alapján kiválasztjuk a következő számot
        # Kiválasztjuk a legmagasabb indexet, amely aktív
        possible_numbers = [i + min_num for i, val in enumerate(current_state) if val == 1]
        if possible_numbers:
            next_num = max(possible_numbers)
        else:
            # Ha nincs aktív szám, válasszuk a leggyakoribb számot
            next_num = most_common_numbers[pos]

        predicted_numbers.append(next_num)

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
