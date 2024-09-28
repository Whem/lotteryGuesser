# neural_network_evolutionary_optimization.py

import numpy as np
from typing import List
from algorithms.models import lg_lottery_winner_number, lg_lottery_type
import random  # Hiányzott a importálás


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1)
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        return self.z2


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    # Lekérdezzük a múltbeli húzásokat és listává alakítjuk
    past_draws_queryset = lg_lottery_winner_number.objects.filter(
        lottery_type=lottery_type_instance
    ).values_list('lottery_type_number', flat=True)

    past_draws = list(past_draws_queryset)  # Konvertálás listává
    print(f"Total past_draws retrieved: {len(past_draws)}")

    # Ellenőrizzük, hogy minden húzás listája és a megfelelő számú számot tartalmazza
    past_draws = [
        draw for draw in past_draws
        if isinstance(draw, list) and len(draw) == lottery_type_instance.pieces_of_draw_numbers
    ]
    print(f"Valid past_draws after filtering: {len(past_draws)}")

    if len(past_draws) < 50:
        selected_numbers = random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        )
        selected_numbers = [int(num) for num in selected_numbers]  # Konvertálás Python int-re
        selected_numbers.sort()
        print(f"Not enough past draws. Selected random numbers: {selected_numbers}")
        return selected_numbers

    window_size = 10
    pieces_of_draw_numbers = lottery_type_instance.pieces_of_draw_numbers
    input_size = window_size * pieces_of_draw_numbers  # Beállítjuk az input_size-t a tényleges méretre
    hidden_size = 20
    output_size = pieces_of_draw_numbers

    population_size = 50
    generations = 100

    population = [SimpleNN(input_size, hidden_size, output_size) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = []
        for nn in population:
            predictions = []
            for i in range(len(past_draws) - window_size):
                input_window = past_draws[i:i + window_size]
                # Ellenőrizzük, hogy minden húzás listája megegyezik-e a szintaxissal
                if not all(isinstance(draw, list) and len(draw) == pieces_of_draw_numbers for draw in input_window):
                    print(f"Skipping invalid draw window starting at index {i}")
                    continue
                input_data = np.array(input_window).flatten()
                # Ellenőrizni kell, hogy input_data mérete megegyezik-e az input_size-val
                if input_data.shape[0] != input_size:
                    print(f"Skipping invalid input_data shape: {input_data.shape}")
                    continue
                output = nn.forward(input_data)
                predictions.append(output)

            if not predictions:
                fitness_scores.append(0)
                continue

            # Az aktuális és predikált számok összehasonlítása
            score = 0
            for j, pred in enumerate(predictions):
                actual = np.array(past_draws[j + window_size])
                pred_rounded = np.round(pred).astype(int)
                # Ellenőrizzük, hogy az actual is megfelelő méretű-e
                if actual.shape[0] != output_size:
                    print(f"Skipping invalid actual draw at index {j + window_size}: {actual}")
                    continue
                # Számok abszolút különbségének összeadása
                score += np.sum(np.abs(actual - pred_rounded))
            fitness_scores.append(1 / (score + 1e-6))

        # Select top performers (10 legjobbak)
        if len(fitness_scores) < 10:
            print("Not enough fitness scores to select top performers.")
            top_performers_indices = np.argsort(fitness_scores)
        else:
            top_performers_indices = np.argsort(fitness_scores)[-10:]
        top_performers = [population[i] for i in top_performers_indices]

        # Create new population by breeding
        new_population = top_performers.copy()
        while len(new_population) < population_size:
            if len(top_performers) < 2:
                parent1 = parent2 = random.choice(top_performers)
            else:
                parent1, parent2 = np.random.choice(top_performers, 2, replace=False)
            child = SimpleNN(input_size, hidden_size, output_size)
            child.w1 = (parent1.w1 + parent2.w1) / 2 + np.random.randn(*child.w1.shape) * 0.1
            child.w2 = (parent1.w2 + parent2.w2) / 2 + np.random.randn(*child.w2.shape) * 0.1
            new_population.append(child)

        population = new_population
        print(f"Generation {generation + 1}/{generations} completed.")

    # Kiválasztjuk a legjobb neurális hálózatot
    if not fitness_scores:
        print("No fitness scores available. Selecting a random network.")
        best_nn = random.choice(population)
    else:
        best_nn_index = np.argmax(fitness_scores)
        best_nn = population[best_nn_index]

    # Utolsó húzás előkészítése predikcióhoz
    last_input_window = past_draws[-window_size:]
    if not all(isinstance(draw, list) and len(draw) == pieces_of_draw_numbers for draw in last_input_window):
        print("Invalid last input window.")
        return random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        )

    last_input = np.array(last_input_window).flatten()
    if last_input.shape[0] != input_size:
        print(f"Invalid last_input shape: {last_input.shape}")
        return random.sample(
            range(lottery_type_instance.min_number, lottery_type_instance.max_number + 1),
            lottery_type_instance.pieces_of_draw_numbers
        )
    prediction = best_nn.forward(last_input)

    # Predikált számok kiválasztása
    predicted_numbers = set()
    for num in np.round(prediction).astype(int):
        if lottery_type_instance.min_number <= num <= lottery_type_instance.max_number:
            predicted_numbers.add(int(num))  # Konvertálás Python int-re

    # Hiányzó számok pótlása véletlenszerűen
    while len(predicted_numbers) < lottery_type_instance.pieces_of_draw_numbers:
        predicted_numbers.add(random.randint(lottery_type_instance.min_number, lottery_type_instance.max_number))

    return sorted(predicted_numbers)
