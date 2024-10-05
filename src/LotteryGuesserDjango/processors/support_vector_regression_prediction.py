# support_vector_regression_prediction.py

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat támogatott vektor regresszió (SVR) segítségével.

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

    if len(past_draws) < 20:
        # Ha nincs elég adat, visszaadjuk a legkisebb 'total_numbers' számot
        selected_numbers = list(range(min_num, min_num + total_numbers))
        return selected_numbers

    # Átalakítjuk a múltbeli húzásokat egy numpy mátrixba
    draw_matrix = np.array(past_draws)

    # Skálázó inicializálása
    scaler = StandardScaler()

    predicted_numbers = []

    # Minden pozícióra külön SVR modellt tanítunk
    for i in range(total_numbers):
        # Készítjük a bemeneteket (X) és a célokat (y)
        X = np.arange(len(draw_matrix)).reshape(-1, 1)  # Idő index
        y = draw_matrix[:, i]

        # Skálázzuk az y értékeket
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # SVR modell inicializálása és tanítása
        model = SVR(kernel='rbf')
        model.fit(X, y_scaled)

        # Következő időpont előrejelzése
        next_index = np.array([[len(draw_matrix)]])
        y_pred_scaled = model.predict(next_index)

        # Visszaskálázás az eredeti tartományba
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Kerekítés és tartományhoz igazítás
        predicted_number = int(round(y_pred[0]))
        if predicted_number < min_num:
            predicted_number = min_num
        elif predicted_number > max_num:
            predicted_number = max_num

        predicted_numbers.append(predicted_number)

    # Egyedivé tesszük a számokat
    predicted_numbers = list(set(predicted_numbers))

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_draws for number in draw]
        number_counts = {}
        for number in all_numbers:
            number_counts[number] = number_counts.get(number, 0) + 1
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        for num, _ in sorted_numbers:
            if num not in predicted_numbers:
                predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Rendezés és levágás a szükséges hosszra
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
