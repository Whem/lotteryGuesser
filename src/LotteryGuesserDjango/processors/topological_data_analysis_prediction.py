# topological_data_analysis_prediction.py

import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import gudhi as gd
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Topological Data Analysis (TDA) alapú elemzéssel.

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

    # Számok gyakoriságának számolása
    all_numbers = [number for draw in past_draws for number in draw]
    number_counts = Counter(all_numbers)
    most_common_numbers = [num for num, count in number_counts.most_common()]

    # Adatok előkészítése a TDA-hoz
    # Minden húzás egy pont a többdimenziós térben
    data = np.array(past_draws)

    # Persistent Homology számítása
    rips_complex = gd.RipsComplex(points=data, max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    # Persistencia diagramok feldolgozása
    # Kivonjuk a 1. dimenzió (hurok) persistenciáját
    loops = [p for p in persistence if p[0] == 1]

    # Feature-ek kinyerése a persistencia diagramból
    # Például a legtöbb hurok, átlagos persistencia, stb.
    num_loops = len(loops)
    avg_persistence = np.mean([p[1][1] - p[1][0] for p in loops]) if loops else 0
    max_persistence = np.max([p[1][1] - p[1][0] for p in loops]) if loops else 0

    # Feature vektor létrehozása
    features = np.array([num_loops, avg_persistence, max_persistence]).reshape(1, -1)

    # Főkomponens Analízis (PCA) a dimenziócsökkentéshez
    pca = PCA(n_components=2)
    past_features = []
    for p in persistence:
        if p[0] == 1:
            loop_persistence = p[1][1] - p[1][0]
            past_features.append([loop_persistence])
    if len(past_features) >= 2:
        pca.fit(past_features)
        reduced_features = pca.transform(features)
    else:
        reduced_features = features  # Ha nincs elég adat, ne csökkentsük

    # Előrejelzés a features alapján
    # Itt egyszerűen példaként a legtöbb hurok alapján választunk számot
    # További fejlesztések: gépi tanulási modellek integrálása a features alapján
    if num_loops > 5:
        # Ha sok hurok van, válasszuk a leggyakoribb számokat
        predicted_numbers = most_common_numbers[:total_numbers]
    elif num_loops > 3:
        # Mérsékelt hurok
        predicted_numbers = most_common_numbers[:total_numbers]
    else:
        # Kevesebb hurok, válasszuk a leggyakoribb számokat
        predicted_numbers = most_common_numbers[:total_numbers]

    # Biztosítjuk, hogy a számok egyediek és az érvényes tartományba esnek
    predicted_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

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
