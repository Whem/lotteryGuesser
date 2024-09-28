# kmeans_clustering_prediction.py

import random
import numpy as np
from sklearn.cluster import KMeans
from algorithms.models import lg_lottery_winner_number

def get_numbers(lottery_type_instance):
    """
    Generates lottery numbers using K-Means clustering.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A sorted list of predicted lottery numbers.
    """
    min_num = lottery_type_instance.min_number
    max_num = lottery_type_instance.max_number
    total_numbers = lottery_type_instance.pieces_of_draw_numbers

    # Retrieve past winning numbers
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).order_by('id').values_list('lottery_type_number', flat=True)

    past_draws = []
    for draw in past_draws_queryset:
        if isinstance(draw, list) and len(draw) == total_numbers:
            past_draws.append(draw)

    if len(past_draws) < 10:
        # Not enough data to perform clustering
        selected_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
        selected_numbers.sort()
        return selected_numbers

    # Prepare data for clustering
    data = []
    for draw in past_draws:
        data.extend(draw)
    data = np.array(data).reshape(-1, 1)

    # Perform K-Means clustering
    k = total_numbers  # Number of clusters equal to numbers to select
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()
    predicted_numbers = [int(round(center)) for center in centers]
    predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

    # Ensure unique numbers
    predicted_numbers = list(set(predicted_numbers))

    # If not enough numbers, fill with random numbers
    if len(predicted_numbers) < total_numbers:
        all_numbers = set(range(min_num, max_num + 1))
        remaining_numbers = list(all_numbers - set(predicted_numbers))
        random.shuffle(remaining_numbers)
        predicted_numbers.extend(remaining_numbers[:total_numbers - len(predicted_numbers)])
    else:
        predicted_numbers = predicted_numbers[:total_numbers]

    # Sort and return the numbers
    predicted_numbers.sort()
    return predicted_numbers
