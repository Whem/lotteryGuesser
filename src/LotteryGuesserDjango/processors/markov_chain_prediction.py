# markov_chain_prediction.py

from collections import defaultdict, Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Markov Chain alapú elemzéssel.

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

    # Készítsünk átmeneti mátrixot (transition matrix)
    transition_counts = defaultdict(Counter)

    for draw in past_draws:
        for i in range(len(draw) - 1):
            current_num = draw[i]
            next_num = draw[i + 1]
            transition_counts[current_num][next_num] += 1

    # Számítsuk ki az átmeneti valószínűségeket
    transition_probs = {}
    for current_num, counter in transition_counts.items():
        total_transitions = sum(counter.values())
        transition_probs[current_num] = {num: count / total_transitions for num, count in counter.items()}

    # Utolsó húzás számainak felhasználása az előrejelzéshez
    last_draw = past_draws[-1]
    predicted_candidates = []

    for num in last_draw:
        if num in transition_probs:
            # Válasszuk a legvalószínűbb következő számot
            next_num = max(transition_probs[num].items(), key=lambda x: x[1])[0]
            predicted_candidates.append(next_num)

    # Eltávolítjuk a duplikátumokat
    predicted_numbers = list(dict.fromkeys(predicted_candidates))

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_draws for number in draw]
        number_counts = Counter(all_numbers)
        most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]

        for num in most_common_numbers:
            predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Biztosítjuk, hogy a számok egyediek és az érvényes tartományba esnek
    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # Ha még mindig kevesebb számunk van, mint szükséges, adjunk hozzá véletlenszerű számokat determinisztikus módon
    if len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(num)
                if len(predicted_numbers) == total_numbers:
                    break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()

    return predicted_numbers
