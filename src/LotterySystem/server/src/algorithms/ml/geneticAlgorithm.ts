import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class GeneticAlgorithm extends BaseAlgorithm {
  name = 'genetic_algorithm';
  displayName = 'Genetic Algorithm';
  description = 'Evolves optimal number combinations through selection and mutation';
  category = 'MACHINE_LEARNING' as const;
  complexity = 'ADVANCED' as const;
  isPremium = true;

  private readonly populationSize = 50;
  private readonly generations = 20;
  private readonly mutationRate = 0.1;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Initialize population
    let population = this.initializePopulation(config);
    
    // Evolution loop
    for (let gen = 0; gen < this.generations; gen++) {
      // Evaluate fitness
      const fitness = population.map(individual => ({
        numbers: individual,
        fitness: this.evaluateFitness(individual, history),
      }));
      
      // Sort by fitness
      fitness.sort((a, b) => b.fitness - a.fitness);
      
      // Selection and reproduction
      const newPopulation: number[][] = [];
      
      // Keep top performers
      const eliteCount = Math.floor(this.populationSize * 0.1);
      for (let i = 0; i < eliteCount; i++) {
        newPopulation.push([...fitness[i].numbers]);
      }
      
      // Create offspring
      while (newPopulation.length < this.populationSize) {
        const parent1 = this.tournamentSelect(fitness);
        const parent2 = this.tournamentSelect(fitness);
        const offspring = this.crossover(parent1, parent2, config);
        this.mutate(offspring, config);
        newPopulation.push(offspring);
      }
      
      population = newPopulation;
    }
    
    // Get best individual
    const finalFitness = population.map(individual => ({
      numbers: individual,
      fitness: this.evaluateFitness(individual, history),
    }));
    finalFitness.sort((a, b) => b.fitness - a.fitness);
    
    const finalMain = finalFitness[0].numbers.sort((a, b) => a - b);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.evolveAdditional(config, history);
    }

    return {
      main: finalMain,
      additional,
      confidence: 0.74,
      executionTime: performance.now() - startTime,
    };
  }

  private initializePopulation(config: LotteryConfig): number[][] {
    const population: number[][] = [];
    
    for (let i = 0; i < this.populationSize; i++) {
      population.push(this.generateRandomNumbers(
        config.minNumber,
        config.maxNumber,
        config.numbersCount
      ));
    }
    
    return population;
  }

  private evaluateFitness(numbers: number[], history: HistoricalDraw[]): number {
    let fitness = 0;
    
    // Reward for matching historical patterns
    history.slice(0, 50).forEach((draw, index) => {
      const weight = 1 - (index / 50);
      const matches = numbers.filter(n => draw.numbers.includes(n)).length;
      fitness += matches * weight;
    });
    
    // Reward for good sum range
    const sum = numbers.reduce((a, b) => a + b, 0);
    const avgHistoricalSum = history.slice(0, 20).reduce((acc, d) => 
      acc + d.numbers.reduce((a, b) => a + b, 0), 0) / 20;
    const sumDiff = Math.abs(sum - avgHistoricalSum);
    fitness -= sumDiff / 100;
    
    // Reward for good distribution (spread)
    const sorted = [...numbers].sort((a, b) => a - b);
    let totalSpread = 0;
    for (let i = 1; i < sorted.length; i++) {
      totalSpread += sorted[i] - sorted[i - 1];
    }
    const avgSpread = totalSpread / (sorted.length - 1);
    fitness += avgSpread / 10;
    
    return fitness;
  }

  private tournamentSelect(
    fitness: { numbers: number[]; fitness: number }[]
  ): number[] {
    const tournamentSize = 5;
    const tournament: typeof fitness = [];
    
    for (let i = 0; i < tournamentSize; i++) {
      const index = Math.floor(Math.random() * fitness.length);
      tournament.push(fitness[index]);
    }
    
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0].numbers;
  }

  private crossover(
    parent1: number[],
    parent2: number[],
    config: LotteryConfig
  ): number[] {
    const offspring = new Set<number>();
    
    // Take half from each parent
    const halfSize = Math.ceil(config.numbersCount / 2);
    
    const shuffled1 = [...parent1].sort(() => Math.random() - 0.5);
    const shuffled2 = [...parent2].sort(() => Math.random() - 0.5);
    
    shuffled1.slice(0, halfSize).forEach(n => offspring.add(n));
    shuffled2.slice(0, halfSize).forEach(n => offspring.add(n));
    
    // Fill remaining if needed
    while (offspring.size < config.numbersCount) {
      const num = Math.floor(Math.random() * (config.maxNumber - config.minNumber + 1)) + config.minNumber;
      offspring.add(num);
    }
    
    return Array.from(offspring).slice(0, config.numbersCount);
  }

  private mutate(numbers: number[], config: LotteryConfig): void {
    for (let i = 0; i < numbers.length; i++) {
      if (Math.random() < this.mutationRate) {
        // Replace with random number
        let newNum: number;
        do {
          newNum = Math.floor(Math.random() * (config.maxNumber - config.minNumber + 1)) + config.minNumber;
        } while (numbers.includes(newNum));
        numbers[i] = newNum;
      }
    }
  }

  private evolveAdditional(config: LotteryConfig, history: HistoricalDraw[]): number[] {
    let population: number[][] = [];
    
    for (let i = 0; i < 20; i++) {
      population.push(this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount!
      ));
    }
    
    for (let gen = 0; gen < 10; gen++) {
      const fitness = population.map(nums => ({
        numbers: nums,
        fitness: history.slice(0, 30).reduce((acc, d) => {
          const matches = nums.filter(n => d.additionalNumbers?.includes(n)).length;
          return acc + matches;
        }, 0),
      }));
      
      fitness.sort((a, b) => b.fitness - a.fitness);
      population = fitness.slice(0, 10).map(f => f.numbers);
      
      while (population.length < 20) {
        population.push(this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount!
        ));
      }
    }
    
    return population[0].sort((a, b) => a - b);
  }
}
