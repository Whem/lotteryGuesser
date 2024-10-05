# bayesian_prediction.py

from collections import defaultdict
from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Bayesiánus valószínűség-alapú elemzéssel.

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

    # Összesített számlálók létrehozása
    number_counts = defaultdict(int)
    total_draws = len(past_draws)

    for draw in past_draws:
        for number in draw:
            number_counts[number] += 1

    # Prior eloszlás: Egyenletes eloszlás, ha nincs előzetes információ
    prior = {number: 1 / (max_num - min_num + 1) for number in range(min_num, max_num + 1)}

    # Likelihood: Múltbeli gyakoriságok alapján számított valószínűségek
    likelihood = {}
    for number in range(min_num, max_num + 1):
        likelihood[number] = number_counts[number] / (total_draws * total_numbers)

    # Posterior valószínűség számítása Bayesiánus szabály alapján
    posterior = {}
    for number in range(min_num, max_num + 1):
        posterior[number] = prior[number] * likelihood[number]

    # Normalizálás
    total_posterior = sum(posterior.values())
    if total_posterior == 0:
        # Elkerüljük a nullával való osztást, ha nincs adat
        posterior = prior
        total_posterior = sum(posterior.values())

    for number in posterior:
        posterior[number] /= total_posterior

    # Számok rendezése a posterior valószínűségük szerint csökkenő sorrendben
    sorted_numbers = sorted(posterior.items(), key=lambda x: x[1], reverse=True)

    # Kiválasztjuk az első 'total_numbers' számot
    predicted_numbers = [num for num, prob in sorted_numbers[:total_numbers]]

    # Biztosítjuk, hogy a számok egyediek és az érvényes tartományba esnek
    predicted_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        # Rendezett gyakoriságok
        most_common = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in most_common:
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
