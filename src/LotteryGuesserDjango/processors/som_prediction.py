# som_prediction.py

import numpy as np
from collections import Counter
from minisom import MiniSom
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance, som_size=(10, 10), sigma=1.0, learning_rate=0.5, random_seed=42):
    """
    Generál lottószámokat Self-Organizing Map (SOM) alapú elemzéssel.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.
    - som_size: A SOM hálózat mérete.
    - sigma: A környezet nagysága.
    - learning_rate: A tanulási ráta.
    - random_seed: Véletlenszerűség kezelése.

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

    # Egyesítsük az összes húzás számát egy NumPy tömbbe
    data = np.array(past_draws)

    # Inicializáljuk a SOM modellt
    som = MiniSom(som_size[0], som_size[1], total_numbers, sigma=sigma, learning_rate=learning_rate,
                 random_seed=random_seed)
    som.train_random(data, 1000)

    # Kiválasztjuk a leggyakoribb neuronokat
    win_map = som.win_map(data)
    frequent_neurons = sorted(win_map.keys(), key=lambda x: len(win_map[x]), reverse=True)

    predicted_numbers = []
    for neuron in frequent_neurons:
        draws = win_map[neuron]
        for draw in draws:
            for num in draw:
                if num not in predicted_numbers and min_num <= num <= max_num:
                    predicted_numbers.append(int(num))  # Konvertálás int-re
                    if len(predicted_numbers) == total_numbers:
                        break
            if len(predicted_numbers) == total_numbers:
                break
        if len(predicted_numbers) == total_numbers:
            break

    # Ha kevesebb szám van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]

        for num in most_common_numbers:
            predicted_numbers.append(int(num))  # Konvertálás int-re
            if len(predicted_numbers) == total_numbers:
                break

    # Biztosítjuk, hogy a számok egyediek és a megengedett tartományba esnek
    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # Ha még mindig kevesebb számunk van, mint szükséges, adjunk hozzá determinisztikus módon
    if len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(int(num))  # Konvertálás int-re
                if len(predicted_numbers) == total_numbers:
                    break

    # Végső rendezés és konvertálás standard Python int típusra
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers = [int(num) for num in predicted_numbers]
    predicted_numbers.sort()

    return predicted_numbers
