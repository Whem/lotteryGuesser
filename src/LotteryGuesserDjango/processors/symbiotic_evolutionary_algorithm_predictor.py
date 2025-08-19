# symbiotic_evolutionary_algorithm_predictor.py

import random
import math
import statistics
from typing import List, Tuple, Set, Dict
from collections import defaultdict
from algorithms.models import lg_lottery_winner_number, lg_lottery_type

class LotteryOrganism:
    def __init__(self, genome: List[int], min_num: int, max_num: int):
        self.genome = genome
        self.fitness = 0
        self.min_num = min_num
        self.max_num = max_num

    def mutate(self, mutation_rate: float) -> List[int]:
        """
        Mutates the organism's genome based on the mutation rate.

        Parameters:
        - mutation_rate: Probability of each gene mutating.

        Returns:
        - A new mutated genome list.
        """
        return [
            random.randint(self.min_num, self.max_num) if random.random() < mutation_rate else gene
            for gene in self.genome
        ]


def create_initial_population(pop_size: int, genome_size: int, min_num: int, max_num: int) -> List[LotteryOrganism]:
    """
    Creates the initial population of organisms.

    Parameters:
    - pop_size: Number of organisms in the population.
    - genome_size: Number of genes in each genome.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.

    Returns:
    - A list of LotteryOrganism instances.
    """
    return [
        LotteryOrganism([random.randint(min_num, max_num) for _ in range(genome_size)], min_num, max_num)
        for _ in range(pop_size)
    ]


def fitness_function(organism: LotteryOrganism, past_draws: List[List[int]]) -> float:
    """
    Calculates the fitness of an organism based on past draws with randomization.

    Parameters:
    - organism: The LotteryOrganism instance.
    - past_draws: List of past lottery number draws.

    Returns:
    - A float representing the fitness score.
    """
    fitness = 0
    for draw in past_draws:
        matches = len(set(organism.genome) & set(draw))
        fitness += matches ** 2  # Quadratic reward for matches
    
    # Add small random noise to avoid identical fitness scores
    noise = random.uniform(-0.1, 0.1)
    return fitness + noise


def symbiotic_crossover(parent1: LotteryOrganism, parent2: LotteryOrganism) -> Tuple[LotteryOrganism, LotteryOrganism]:
    """
    Performs symbiotic crossover between two parent organisms.

    Parameters:
    - parent1: The first parent LotteryOrganism.
    - parent2: The second parent LotteryOrganism.

    Returns:
    - A tuple containing two child LotteryOrganism instances.
    """
    if len(parent1.genome) != len(parent2.genome):
        raise ValueError("Parents must have genomes of the same length.")

    crossover_point = random.randint(1, len(parent1.genome) - 1)
    child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
    child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
    return (
        LotteryOrganism(child1_genome, parent1.min_num, parent1.max_num),
        LotteryOrganism(child2_genome, parent1.min_num, parent1.max_num)
    )


def evolve_population(population: List[LotteryOrganism], past_draws: List[List[int]],
                     generations: int = 50, mutation_rate: float = 0.1) -> List[LotteryOrganism]:
    """
    Evolves the population over a number of generations.

    Parameters:
    - population: The current population of LotteryOrganism instances.
    - past_draws: List of past lottery number draws.
    - generations: Number of generations to evolve.
    - mutation_rate: Probability of each gene mutating.

    Returns:
    - The evolved population as a list of LotteryOrganism instances.
    """
    for generation in range(generations):
        # Evaluate fitness
        for organism in population:
            organism.fitness = fitness_function(organism, past_draws)

        # Sort population by fitness in descending order
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Select top performers (elite)
        elite_size = max(2, int(len(population) * 0.1))
        elite = population[:elite_size]

        # Initialize new population with elites
        new_population = elite.copy()

        # Generate offspring through crossover and mutation
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(elite, 2)
            child1, child2 = symbiotic_crossover(parent1, parent2)
            child1.genome = child1.mutate(mutation_rate)
            child2.genome = child2.mutate(mutation_rate)
            new_population.extend([child1, child2])

        # Truncate to maintain population size
        population = new_population[:len(population)]

    return population


def calculate_field_weights(
        past_draws: List[List[int]],
        min_num: int,
        max_num: int,
        excluded_numbers: Set[int]
) -> Dict[int, float]:
    """
    Calculate weights based on statistical similarity to past draws.

    This function assigns weights to each number based on how closely it aligns with the
    statistical properties (mean) of historical data. Numbers not yet selected are given higher
    weights if they are more statistically probable based on the average mean.

    Parameters:
    - past_draws: List of past lottery number draws.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - excluded_numbers: Set of numbers to exclude from selection.

    Returns:
    - A dictionary mapping each number to its calculated weight.
    """
    weights = defaultdict(float)

    if not past_draws:
        return weights

    try:
        # Calculate overall mean from past draws
        all_numbers = [num for draw in past_draws for num in draw]
        overall_mean = statistics.mean(all_numbers)
        overall_stdev = statistics.stdev(all_numbers) if len(all_numbers) > 1 else 1.0

        for num in range(min_num, max_num + 1):
            if num in excluded_numbers:
                continue

            # Calculate z-score for the number
            z_score = (num - overall_mean) / overall_stdev if overall_stdev != 0 else 0.0

            # Assign higher weight to numbers closer to the mean
            weight = max(0, 1 - abs(z_score))
            weights[num] = weight

    except Exception as e:
        print(f"Weight calculation error: {e}")
        # Fallback to uniform weights
        for num in range(min_num, max_num + 1):
            if num not in excluded_numbers:
                weights[num] = 1.0

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        for num in weights:
            weights[num] /= total_weight

    return weights


def weighted_random_choice(weights: Dict[int, float], available_numbers: Set[int]) -> int:
    """
    Selects a random number based on weighted probabilities.

    Parameters:
    - weights: A dictionary mapping numbers to their weights.
    - available_numbers: A set of numbers available for selection.

    Returns:
    - A single selected number.
    """
    try:
        numbers = list(available_numbers)
        number_weights = [weights.get(num, 1.0) for num in numbers]
        total = sum(number_weights)
        if total == 0:
            return random.choice(numbers)
        probabilities = [w / total for w in number_weights]
        selected = random.choices(numbers, weights=probabilities, k=1)[0]
        return selected
    except Exception as e:
        print(f"Weighted random choice error: {e}")
        return random.choice(list(available_numbers)) if available_numbers else None


def generate_random_numbers(lottery_type_instance: lg_lottery_type, min_num: int, max_num: int,
                           total_numbers: int) -> List[int]:
    """
    Generate random numbers as a fallback mechanism.

    This function generates random numbers while ensuring they fall within the specified
    range and are unique.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.
    - min_num: Minimum number in the lottery range.
    - max_num: Maximum number in the lottery range.
    - total_numbers: Total numbers to generate.

    Returns:
    - A sorted list of randomly generated lottery numbers.
    """
    numbers = set()
    required_numbers = total_numbers

    while len(numbers) < required_numbers:
        num = random.randint(min_num, max_num)
        numbers.add(num)

    return sorted(list(numbers))


def get_numbers(lottery_type_instance: lg_lottery_type) -> Tuple[List[int], List[int]]:
    """
    Generates lottery numbers using a Symbiotic Evolutionary Algorithm.

    This function generates both main numbers and additional numbers (if applicable) by analyzing
    past lottery draws using a symbiotic evolutionary algorithm. It evolves a population of
    organisms to maximize fitness based on matches with past draws and ensures that the generated
    numbers are unique and within the specified range.

    Parameters:
    - lottery_type_instance: An instance of lg_lottery_type model.

    Returns:
    - A tuple containing two lists:
        - main_numbers: A sorted list of predicted main lottery numbers.
        - additional_numbers: A sorted list of predicted additional lottery numbers (if applicable).
    """
    try:
        min_num = int(lottery_type_instance.min_number)
        max_num = int(lottery_type_instance.max_number)
        total_numbers = int(lottery_type_instance.pieces_of_draw_numbers)

        # Retrieve past winning numbers with randomization
        past_draws_queryset = lg_lottery_winner_number.objects.filter(
            lottery_type=lottery_type_instance
        ).order_by('-id').values_list('lottery_type_number', flat=True)

        all_past_draws = [
            draw for draw in past_draws_queryset
            if isinstance(draw, list) and len(draw) == total_numbers
        ]
        
        # Randomize the selection of past draws for variety
        if len(all_past_draws) > 20:
            # Use random sample of past draws (70-90% of available data)
            sample_size = max(20, int(len(all_past_draws) * (0.7 + random.random() * 0.2)))
            past_draws = random.sample(all_past_draws, min(sample_size, len(all_past_draws)))
        else:
            past_draws = all_past_draws

        if len(past_draws) < 10:
            # If not enough past draws, generate random main numbers
            main_numbers = random.sample(range(min_num, max_num + 1), total_numbers)
            main_numbers.sort()
        else:
            # Create initial population
            population_size = 100
            population = create_initial_population(population_size, total_numbers, min_num, max_num)

            # Evolve population
            evolved_population = evolve_population(population, past_draws)

            # Select from top organisms with randomization
            sorted_population = sorted(evolved_population, key=lambda x: x.fitness, reverse=True)
            top_count = max(1, len(sorted_population) // 10)  # Top 10% of population
            best_organism = random.choice(sorted_population[:top_count])

            # Ensure uniqueness and correct range
            predicted_numbers = list(set(best_organism.genome))
            predicted_numbers = [num for num in predicted_numbers if min_num <= num <= max_num]

            # If not enough unique numbers, fill with weighted random selection
            if len(predicted_numbers) < total_numbers:
                weights = calculate_field_weights(
                    past_draws=past_draws,
                    min_num=min_num,
                    max_num=max_num,
                    excluded_numbers=set(predicted_numbers)
                )

                while len(predicted_numbers) < total_numbers:
                    number = weighted_random_choice(weights, set(range(min_num, max_num + 1)) - set(predicted_numbers))
                    if number is not None:
                        predicted_numbers.append(number)

            # Sort and trim to required number of numbers
            main_numbers = sorted(predicted_numbers)[:total_numbers]

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            # Generate additional numbers using the same approach
            additional_min_num = int(lottery_type_instance.additional_min_number)
            additional_max_num = int(lottery_type_instance.additional_max_number)
            additional_total_numbers = int(lottery_type_instance.additional_numbers_count)

            # Retrieve past additional numbers with randomization
            past_additional_draws_queryset = lg_lottery_winner_number.objects.filter(
                lottery_type=lottery_type_instance
            ).order_by('-id').values_list('additional_numbers', flat=True)

            all_past_additional_draws = [
                draw for draw in past_additional_draws_queryset
                if isinstance(draw, list) and len(draw) == additional_total_numbers
            ]
            
            # Randomize the selection of past additional draws
            if len(all_past_additional_draws) > 10:
                sample_size = max(10, int(len(all_past_additional_draws) * (0.7 + random.random() * 0.2)))
                past_additional_draws = random.sample(all_past_additional_draws, min(sample_size, len(all_past_additional_draws)))
            else:
                past_additional_draws = all_past_additional_draws

            if len(past_additional_draws) < 5:
                # If not enough past draws, generate random additional numbers
                additional_numbers = random.sample(range(additional_min_num, additional_max_num + 1), additional_total_numbers)
                additional_numbers.sort()
            else:
                # Create initial population for additional numbers
                population_size = 50
                population = create_initial_population(population_size, additional_total_numbers,
                                                      additional_min_num, additional_max_num)

                # Evolve population for additional numbers
                evolved_population = evolve_population(population, past_additional_draws, generations=30, mutation_rate=0.05)

                # Select from top organisms with randomization
                sorted_population = sorted(evolved_population, key=lambda x: x.fitness, reverse=True)
                top_count = max(1, len(sorted_population) // 10)  # Top 10% of population
                best_organism = random.choice(sorted_population[:top_count])

                # Ensure uniqueness and correct range
                predicted_additional = list(set(best_organism.genome))
                predicted_additional = [num for num in predicted_additional if additional_min_num <= num <= additional_max_num]

                # If not enough unique numbers, fill with weighted random selection
                if len(predicted_additional) < additional_total_numbers:
                    weights = calculate_field_weights(
                        past_draws=past_additional_draws,
                        min_num=additional_min_num,
                        max_num=additional_max_num,
                        excluded_numbers=set(predicted_additional)
                    )

                    while len(predicted_additional) < additional_total_numbers:
                        number = weighted_random_choice(weights, set(range(additional_min_num, additional_max_num + 1)) - set(predicted_additional))
                        if number is not None:
                            predicted_additional.append(number)

                # Sort and trim to required number of additional numbers
                additional_numbers = sorted(predicted_additional)[:additional_total_numbers]

        return main_numbers, additional_numbers

    except Exception as e:
        # Log the error (consider using a proper logging system)
        print(f"Error in symbiotic_evolutionary_algorithm_predictor: {str(e)}")
        # Fall back to random number generation
        main_numbers = random.sample(range(int(lottery_type_instance.min_number), int(lottery_type_instance.max_number) + 1),
                                     int(lottery_type_instance.pieces_of_draw_numbers))
        main_numbers.sort()

        additional_numbers = []
        if lottery_type_instance.has_additional_numbers:
            additional_numbers = random.sample(range(int(lottery_type_instance.additional_min_number),
                                                    int(lottery_type_instance.additional_max_number) + 1),
                                              int(lottery_type_instance.additional_numbers_count))
            additional_numbers.sort()

        return main_numbers, additional_numbers
