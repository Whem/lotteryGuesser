# association_rule_prediction.py

import numpy as np
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from algorithms.models import lg_lottery_winner_number


def get_numbers(lottery_type_instance, min_support=0.05, metric="confidence", min_threshold=0.6):
    """
    Generál lottószámokat Association Rule Mining alapú elemzéssel.

    Paraméterek:
    - lottery_type_instance: Az lg_lottery_type modell egy példánya.
    - min_support: Minimum támogatottság a frequent itemset-ekhez.
    - metric: A metrika a szabályok kiválasztásához.
    - min_threshold: Minimum küszöbérték a metrika számára.

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

    # Konvertáljuk a húzásokat DataFrame formátumba a mlxtend számára
    df = pd.DataFrame(past_draws)

    # **Javítás:** Konvertáljuk a számokat stringgé, mielőtt a .str accessor-t használjuk
    df = df.stack().astype(str).str.get_dummies().groupby(level=0).sum()

    # Alkalmazzuk az Apriori algoritmust
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        # Ha nincs találat, használjuk a leggyakoribb számokat
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        most_common = number_counts.most_common(total_numbers)
        predicted_numbers = [num for num, count in most_common]
    else:
        # Generáljunk asszociációs szabályokat
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

        # Vegyük az utolsó húzás számait
        last_draw = past_draws[-1]
        predicted_candidates = []

        for num in last_draw:
            # Szűrjük a szabályokat, ahol a jelenlegi szám szerepel a left-hand side-en
            relevant_rules = rules[rules['antecedents'].apply(lambda x: num in x)]
            if not relevant_rules.empty:
                # Válasszuk ki a legmagasabb confidence-ű szabályt
                top_rule = relevant_rules.sort_values(by='confidence', ascending=False).iloc[0]
                next_num = list(top_rule['consequents'])[0]
                predicted_candidates.append(next_num)

        # Eltávolítjuk a duplikátumokat
        predicted_numbers = list(dict.fromkeys(predicted_candidates))

        if len(predicted_numbers) < total_numbers:
            # Kiegészítjük a leggyakoribb számokkal
            all_numbers = [num for draw in past_draws for num in draw]
            number_counts = Counter(all_numbers)
            most_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]

            for num in most_common_numbers:
                predicted_numbers.append(num)
                if len(predicted_numbers) == total_numbers:
                    break

    # Biztosítjuk, hogy a számok egyediek és az érvényes tartományba esnek
    predicted_numbers = [int(num) for num in predicted_numbers if min_num <= num <= max_num]

    # Ha még mindig kevesebb számunk van, mint szükséges, adjunk hozzá determinisztikus módon
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
