# genetic_algorithm_evolutionary_optimization.py

import random
from typing import List
import numpy as np
from deap import base, creator, tools, algorithms
from algorithms.models import lg_lottery_winner_number, lg_lottery_type


def get_numbers(lottery_type_instance: lg_lottery_type) -> List[int]:
    past_draws = list(
        lg_lottery_winner_number.objects.filter(lottery_type=lottery_type_instance).values_list('lottery_type_number',
                                                                                                flat=True))

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, lottery_type_instance.min_number, lottery_type_instance.max_number)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int,
                     n=lottery_type_instance.pieces_of_draw_numbers)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        score = sum(1 for draw in past_draws if len(set(individual) & set(draw)) >= 3)
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=lottery_type_instance.min_number,
                     up=lottery_type_instance.max_number, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=100)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    return sorted(set(best_individual))

