# k_nearest_neighbors_prediction.py

import numpy as np
import random
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Lekérdezzük a múltbeli húzásokat és listává alakítjuk
    past_draws = list(
        lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id').values_list('lottery_type_number', flat=False)
    )

    # Alakítsuk át a húzásokat listává, eltávolítva az extra listát
    # Feltételezzük, hogy minden draw egy tuple, amely tartalmaz egy listát
    # Példa: [([16, 61, 71, 77, 89],), ([1, 49, 64, 67, 71],), ...]

    # Első szintű laposítás: tuple-ből kinyerjük a listát
    past_draws = [draw[0] for draw in past_draws]

    # Ellenőrizzük, hogy a past_draws nem tartalmaz-e további beágyazást
    # Ha igen, akkor laposítsuk tovább
    # Példa: [[[16, 61, 71, 77, 89]], [[1, 49, 64, 67, 71]], ...] -> [[16, 61, 71, 77, 89], [1, 49, 64, 67, 71], ...]

    # Második szintű laposítás, ha szükséges
    flattened_past_draws = []
    for draw in past_draws:
        if isinstance(draw, (list, tuple)) and all(isinstance(num, (int, float)) for num in draw):
            flattened_past_draws.append(draw)
        elif isinstance(draw, (list, tuple)):
            for sub_draw in draw:
                if isinstance(sub_draw, (list, tuple)) and all(isinstance(num, (int, float)) for num in sub_draw):
                    flattened_past_draws.append(sub_draw)
        else:
            raise ValueError("A past_draws adat nem a várt formátumban van.")

    past_draws = flattened_past_draws

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

    # Ellenőrizzük, hogy X 2D legyen
    if X.ndim != 2:
        raise ValueError(f"X dimenziója {X.ndim}, de a StandardScaler 2D-t vár.")

    # Standardizálás
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # X_scaled shape: (n_samples, window_size * numbers_per_draw)

    # Címkézés többcímkézésű problémára
    y_multi = []
    for draw in y:
        label = [0] * (lottery_type_instance.max_number + 1)
        for num in draw:
            if 0 <= num <= lottery_type_instance.max_number:
                label[num] = 1
        y_multi.append(label)

    y_multi = np.array(y_multi)  # Shape: (n_samples, max_number + 1)

    # Osztjuk fel az adatokat tanulásra és tesztelésre
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_multi, test_size=0.2, random_state=42
    )

    # KNN modell létrehozása és tanítása MultiOutputClassifier-rel
    base_model = KNeighborsClassifier(n_neighbors=5)
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
