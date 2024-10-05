# lightgbm_prediction.py

import numpy as np
from collections import Counter
import lightgbm as lgb
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

def get_numbers(lottery_type_instance, num_iterations=100, random_state=42):
    """
    Generál lottószámokat LightGBM alapú elemzéssel.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.
    - num_iterations: A LightGBM tréning iterációinak száma.
    - random_state: Véletlenszerűség kezelése a modellezéshez.

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

    # Adatok előkészítése a LightGBM-hez
    X = np.arange(len(past_draws)).reshape(-1, 1)  # Húzások indexe

    models = []
    for num_position in range(total_numbers):
        y = np.array([draw[num_position] for draw in past_draws])
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'seed': random_state
        }
        model = lgb.train(params, train_data, num_boost_round=num_iterations)
        models.append(model)

    # Predikció a következő húzásra
    next_index = np.array([[len(past_draws)]])
    predicted_numbers = []
    for idx, model in enumerate(models):
        pred = model.predict(next_index)[0]
        pred_int = int(round(pred))
        pred_int = max(min_num, min(pred_int, max_num))
        if pred_int not in predicted_numbers:
            predicted_numbers.append(pred_int)
        if len(predicted_numbers) == total_numbers:
            break

    # Ha kevesebb szám van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]
        for num in most_common_numbers:
            predicted_numbers.append(int(num))
            if len(predicted_numbers) == total_numbers:
                break

    # Biztosítjuk, hogy a számok egyediek és a megengedett tartományba esnek
    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # Ha még mindig kevesebb számunk van, mint szükséges, adjunk hozzá determinisztikus módon
    if len(predicted_numbers) < total_numbers:
        for num in range(min_num, max_num + 1):
            if num not in predicted_numbers:
                predicted_numbers.append(int(num))
                if len(predicted_numbers) == total_numbers:
                    break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()

    return predicted_numbers
