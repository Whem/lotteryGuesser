# graph_centrality_prediction.py
import numpy as np
import networkx as nx
from collections import Counter
from typing import List, Tuple, Dict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """Graph centrality predictor for combined lottery types."""
    main_numbers = generate_number_set(
        lottery_type_instance,
        lottery_type_instance.min_number,
        lottery_type_instance.max_number,
        lottery_type_instance.pieces_of_draw_numbers,
        True
    )

    additional_numbers = []
    if lottery_type_instance.has_additional_numbers:
        additional_numbers = generate_number_set(
            lottery_type_instance,
            lottery_type_instance.additional_min_number,
            lottery_type_instance.additional_max_number,
            lottery_type_instance.additional_numbers_count,
            False
        )

    return main_numbers, additional_numbers


def generate_number_set(
        lottery_type_instance: lg_lottery_type,
        min_num: int,
        max_num: int,
        required_numbers: int,
        is_main: bool
) -> List[int]:
    """Generate numbers using graph centrality analysis."""
    past_draws = get_historical_data(lottery_type_instance, is_main)

    if len(past_draws) < 20:
        return list(range(min_num, min_num + required_numbers))

    # Build transition graph
    G = build_transition_graph(past_draws, min_num, max_num)

    # Get predicted numbers
    predicted_numbers = analyze_graph_centrality(G, required_numbers)

    # Ensure valid numbers
    predicted_numbers = validate_numbers(
        predicted_numbers,
        past_draws,
        min_num,
        max_num,
        required_numbers
    )

    return sorted(predicted_numbers)


def get_historical_data(lottery_type_instance: lg_lottery_type, is_main: bool) -> List[List[int]]:
    """Get historical lottery data."""
    past_draws = list(lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id'))

    if is_main:
        return [draw.lottery_type_number for draw in past_draws
                if isinstance(draw.lottery_type_number, list)]
    else:
        return [draw.additional_numbers for draw in past_draws
                if hasattr(draw, 'additional_numbers') and
                isinstance(draw.additional_numbers, list)]


def build_transition_graph(past_draws: List[List[int]], min_num: int, max_num: int) -> nx.DiGraph:
    """Build transition graph from historical data."""
    G = nx.DiGraph()
    G.add_nodes_from(range(min_num, max_num + 1))

    for draw in past_draws:
        for i in range(len(draw) - 1):
            src, dest = draw[i], draw[i + 1]
            if G.has_edge(src, dest):
                G[src][dest]['weight'] += 1
            else:
                G.add_edge(src, dest, weight=1)

    return G


def analyze_graph_centrality(G: nx.DiGraph, required_numbers: int) -> List[int]:
    """Analyze graph centrality measures."""
    # Check connectivity
    if not nx.is_strongly_connected(G):
        largest_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G

    if len(G_sub) < 2:
        return analyze_pagerank_only(G, required_numbers)

    return analyze_combined_centrality(G_sub, G, required_numbers)


def analyze_pagerank_only(G: nx.DiGraph, required_numbers: int) -> List[int]:
    """Analyze using PageRank when graph is too small."""
    pagerank = nx.pagerank(G, weight='weight')
    sorted_numbers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    return [num for num, rank in sorted_numbers[:required_numbers]]


def analyze_combined_centrality(G_sub: nx.DiGraph, G: nx.DiGraph, required_numbers: int) -> List[int]:
    """Analyze using combined centrality measures."""
    try:
        eigen_centrality = nx.eigenvector_centrality_numpy(G_sub, weight='weight')
    except nx.PowerIterationFailedConvergence:
        eigen_centrality = nx.eigenvector_centrality(
            G_sub,
            weight='weight',
            max_iter=1000,
            tol=1e-06
        )

    pagerank = nx.pagerank(G, weight='weight')

    combined_centrality = {}
    for node in G_sub.nodes():
        eigen_norm = eigen_centrality.get(node, 0)
        pagerank_norm = pagerank.get(node, 0)
        combined_centrality[node] = 0.5 * eigen_norm + 0.5 * pagerank_norm

    sorted_numbers = sorted(
        combined_centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [num for num, _ in sorted_numbers[:required_numbers]]


def validate_numbers(
        predicted_numbers: List[int],
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        required_numbers: int
) -> List[int]:
    """Validate and fill missing numbers."""
    valid_numbers = [
        int(num) for num in predicted_numbers
        if min_num <= num <= max_num
    ]

    if len(valid_numbers) < required_numbers:
        all_numbers = [num for draw in past_draws for num in draw]
        number_counts = Counter(all_numbers)
        common_numbers = [
            num for num, _ in number_counts.most_common()
            if num not in valid_numbers
        ]

        valid_numbers.extend(
            common_numbers[:required_numbers - len(valid_numbers)]
        )

    return valid_numbers[:required_numbers]