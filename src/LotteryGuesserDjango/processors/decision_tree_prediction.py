# decision_tree_prediction.py

import numpy as np
import random
from typing import List
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Lekérdezzük a múltbeli húzásokat és listává alakítjuk
    past_draws = list(
        lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id')[:200].values_list('lottery_type_number', flat=False)
    )

    # Alakítsuk át a húzásokat listává, eltávolítva az extra listát
    past_draws = [draw[0] for draw in past_draws]

    # Készítsük el a training adatokat
    X = []
    y = []
    window_size = 5  # Példa ablak méret

    for i in range(len(past_draws) - window_size):
        window = past_draws[i:i + window_size]
        # Flattening the window (list of lists to single list)
        flat_window = [num for draw in window for num in draw]
        X.append(flat_window)
        y.append(past_draws[i + window_size])

    X = np.array(X)  # Shape: (n_samples, window_size * numbers_per_draw)
    y = np.array(y)  # Shape: (n_samples, numbers_per_draw)

    # Standardizálás
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Should be 2D now

    # Címkézés többcímkézésű problémára
    y_multi = []
    for draw in y:
        label = [0] * (lottery_type_instance.max_number + 1)
        for num in draw:
            if 0 <= num <= lottery_type_instance.max_number:
                label[num] = 1
        y_multi.append(label)

    y_multi = np.array(y_multi)

    # Osztjuk fel az adatokat tanulásra és tesztelésre
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_multi, test_size=0.2, random_state=42
    )

    # Decision Tree modell létrehozása és tanítása
    base_model = DecisionTreeClassifier(random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    # Előrejelzés a teszt adatokon
    predictions = model.predict(X_test)

    # Számok kiválasztása a legmagasabb valószínűségek alapján
    predicted_numbers = set()
    for pred in predictions:
        nums = [i for i, val in enumerate(pred) if val == 1]
        predicted_numbers.update(nums)
        if len(predicted_numbers) >= lottery_type_instance.pieces_of_draw_numbers:
            break

    # Hiányzó számok pótlása véletlenszerűen
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        new_number = random.randint(
            lottery_type_instance.min_number, lottery_type_instance.max_number
        )
        predicted_numbers.add(new_number)

    # Biztosítjuk, hogy a számok a megadott tartományon belül legyenek
    predicted_numbers = [
        num for num in predicted_numbers
        if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number
    ]

    return sorted(predicted_numbers)[:lottery_type_instance.pieces_of_draw_numbers]
