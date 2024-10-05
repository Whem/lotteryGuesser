# graph_centrality_prediction.py

import numpy as np
import networkx as nx
from collections import Counter
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generál lottószámokat Graph Centrality alapú elemzéssel.

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

    # Transition Graph Létrehozása
    G = nx.DiGraph()
    G.add_nodes_from(range(min_num, max_num + 1))

    for draw in past_draws:
        for i in range(len(draw) - 1):
            src = draw[i]
            dest = draw[i + 1]
            if G.has_edge(src, dest):
                G[src][dest]['weight'] += 1
            else:
                G.add_edge(src, dest, weight=1)

    # Ellenőrizzük, hogy a gráf erősen kapcsolt-e
    if not nx.is_strongly_connected(G):
        # Ha nem, válasszuk a legnagyobb erősen kapcsolt komponens
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G

    # Ellenőrizzük, hogy a kiválasztott subgraph megfelelő méretű-e
    if len(G_sub) < 2:
        # Ha túl kicsi, használjuk csak a PageRank-et
        pagerank = nx.pagerank(G, weight='weight')
        sorted_numbers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        predicted_numbers = [num for num, rank in sorted_numbers[:total_numbers]]
    else:
        # Központi Mérőszámok Számítása
        try:
            eigen_centrality = nx.eigenvector_centrality_numpy(G_sub, weight='weight')
        except nx.PowerIterationFailedConvergence:
            # Növeljük az iterációk számát és csökkentsük a toleranciát
            eigen_centrality = nx.eigenvector_centrality(
                G_sub, weight='weight', max_iter=1000, tol=1e-06
            )

        # PageRank Számítása
        pagerank = nx.pagerank(G, weight='weight')

        # Kombinált Központi Mérőszámok
        combined_centrality = {}
        for node in G_sub.nodes():
            eigen_norm = eigen_centrality.get(node, 0)
            pagerank_norm = pagerank.get(node, 0)
            combined_centrality[node] = 0.5 * eigen_norm + 0.5 * pagerank_norm

        # Számok Rangsorolása a Kombinált Centrality alapján
        sorted_numbers = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        predicted_numbers = [num for num, centrality in sorted_numbers[:total_numbers]]

    # Biztosítjuk, hogy a számok egyediek és az érvényes tartományba esnek
    predicted_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

    # Ha kevesebb számunk van, mint szükséges, kiegészítjük a leggyakoribb számokkal
    if len(predicted_numbers) < total_numbers:
        all_numbers = [number for draw in past_draws for number in draw]
        number_counts = Counter(all_numbers)
        sorted_common_numbers = [num for num, count in number_counts.most_common() if num not in predicted_numbers]

        for num in sorted_common_numbers:
            predicted_numbers.append(num)
            if len(predicted_numbers) == total_numbers:
                break

    # Végső rendezés
    predicted_numbers = predicted_numbers[:total_numbers]
    predicted_numbers.sort()
    return predicted_numbers
