# ensemble_prediction.py

from collections import Counter
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

# Import get_numbers functions from existing predictor modules
from .fuzzy_logic_prediction import get_numbers as fuzzy_logic_get_numbers
from .cellular_automaton_prediction import get_numbers as cellular_automaton_get_numbers
from .topological_data_analysis_prediction import get_numbers as tda_get_numbers
from .symbolic_regression_prediction import get_numbers as symbolic_regression_get_numbers
from .graph_centrality_prediction import get_numbers as graph_centrality_get_numbers
from .chaotic_map_prediction import get_numbers as chaotic_map_get_numbers


def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Ensemble alapú elemzéssel.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.

    Visszatérési érték:
    - Egy rendezett lista a megjósolt lottószámokról.
    """
    # Gyűjtsük össze a predikciókat minden modellből
    predictions = []

    predictions.append(fuzzy_logic_get_numbers(lottery_type_instance))
    predictions.append(cellular_automaton_get_numbers(lottery_type_instance))
    predictions.append(tda_get_numbers(lottery_type_instance))
    predictions.append(symbolic_regression_get_numbers(lottery_type_instance))
    predictions.append(graph_centrality_get_numbers(lottery_type_instance))
    predictions.append(chaotic_map_get_numbers(lottery_type_instance))

    # Gyűjtsük össze az összes predikált számot
    all_predicted_numbers = [num for pred in predictions for num in pred]

    # Számoljuk meg a számok előfordulását
    number_counts = Counter(all_predicted_numbers)

    # Válasszuk ki a leggyakoribb számokat a total_numbers mennyiségben
    total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)
    most_common = number_counts.most_common(total_numbers)
    predicted_numbers = [num for num, count in most_common]

    # Biztosítjuk, hogy a számok egyediek és a megengedett tartományon belül maradjanak
    min_num = int(lottery_type_instance.min_number)
    max_num = int(lottery_type_instance.max_number)

    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        # Gyűjtsük össze az összes szám előfordulását
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('id').values_list('lottery_type_number', flat=True)

        past_draws = [
            [int(num) for num in draw] for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]

        all_numbers = [number for draw in past_draws for number in draw]
        number_counts_past = Counter(all_numbers)
        sorted_common_numbers = [num for num, count in number_counts_past.most_common() if num not in predicted_numbers]

        for num in sorted_common_numbers:
            predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()

    return predicted_numbers
