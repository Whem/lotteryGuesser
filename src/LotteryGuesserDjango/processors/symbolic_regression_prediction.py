# symbolic_regression_prediction.py

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Symbolic Regression (Szimbólumikus Regresszió) alapú elemzéssel.

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

    # Elrendezés: Bemenet (X) és kimenet (y) létrehozása
    # Például, ha a múlt 5 húzása alapján szeretnénk előrejelezni a következőt
    window_size = 5  # A múltbeli húzások száma a bemenethez
    X = []
    y = []

    for i in range(len(past_draws) - window_size):
        window = past_draws[i:i + window_size]
        next_draw = past_draws[i + window_size]
        # Flatten the window
        flattened_window = [num for draw in window for num in draw]
        X.append(flattened_window)
        # Célérték: Következő húzás számai
        y.append(next_draw)

    X = np.array(X)
    y = np.array(y)

    # Modellezés minden pozícióra külön
    predicted_numbers = []
    for pos in range(total_numbers):
        # Célérték kiválasztása
        y_pos = y[:, pos]

        # Definiáljuk a fitness függvényt (például MSE)
        def my_fitness(y, y_pred, sample_weight):
            return np.mean((y - y_pred) ** 2)

        mse = make_fitness(function=my_fitness, greater_is_better=False)

        # Symbolic Regressor inicializálása
        est = SymbolicRegressor(
            population_size=1000,
            generations=20,
            tournament_size=20,
            stopping_criteria=0.01,
            const_range=(0, 10),
            init_depth=(2, 6),
            function_set=['add', 'sub', 'mul', 'div'],
            metric=mse,
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            n_jobs=1,
            random_state=0
        )

        # Modell illesztése
        est.fit(X, y_pos)

        # A legjobb modell előrejelzése
        # Használjuk az utolsó window-t az előrejelzéshez
        last_window = past_draws[-window_size:]
        flattened_last_window = np.array([num for draw in last_window for num in draw]).reshape(1, -1)
        predicted_value = est.predict(flattened_last_window)[0]

        # Kerekítés és tartományhoz igazítás
        predicted_number = int(round(predicted_value))
        if predicted_number < min_num:
            predicted_number = min_num
        elif predicted_number > max_num:
            predicted_number = max_num

        predicted_numbers.append(predicted_number)

    # Eltávolítjuk a duplikátumokat
    predicted_numbers = list(dict.fromkeys(predicted_numbers))

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_draws for number in draw]
        number_counts = Counter(all_numbers)
        sorted_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]

        for num in sorted_numbers:
            predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
