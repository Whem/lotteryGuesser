import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class MarkovChain extends BaseAlgorithm {
  name = 'markov_chain';
  displayName = 'Markov Chain';
  description = 'Uses transition probabilities between numbers to predict likely sequences';
  category = 'PROBABILITY' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Build transition matrix
    const transitionMatrix = this.buildTransitionMatrix(history, config.maxNumber);
    
    // Start from most frequent number
    const frequency = this.calculateFrequency(history);
    const sortedByFreq = Array.from(frequency.entries()).sort((a, b) => b[1] - a[1]);
    
    const mainNumbers: number[] = [];
    let currentNumber = sortedByFreq[0]?.[0] || Math.floor((config.maxNumber + config.minNumber) / 2);
    mainNumbers.push(currentNumber);
    
    // Walk the Markov chain
    while (mainNumbers.length < config.numbersCount) {
      const nextNumber = this.selectNextNumber(
        currentNumber,
        transitionMatrix,
        mainNumbers,
        config.minNumber,
        config.maxNumber
      );
      
      if (nextNumber !== null) {
        mainNumbers.push(nextNumber);
        currentNumber = nextNumber;
      } else {
        // Fallback: select random number not yet selected
        const available = [];
        for (let i = config.minNumber; i <= config.maxNumber; i++) {
          if (!mainNumbers.includes(i)) available.push(i);
        }
        if (available.length > 0) {
          const randomNum = available[Math.floor(Math.random() * available.length)];
          mainNumbers.push(randomNum);
          currentNumber = randomNum;
        } else {
          break;
        }
      }
    }
    
    const finalMain = mainNumbers.sort((a, b) => a - b);

    // Additional numbers with simpler approach
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFreq = this.calculateAdditionalFrequency(history);
      additional = this.selectTopN(additionalFreq, config.additionalNumbersCount);
      
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!,
          config.additionalMaxNumber!,
          config.additionalNumbersCount
        );
      }
    }

    return {
      main: finalMain,
      additional,
      confidence: 0.75,
      executionTime: performance.now() - startTime,
    };
  }

  private buildTransitionMatrix(
    history: HistoricalDraw[],
    maxNumber: number
  ): Map<number, Map<number, number>> {
    const matrix = new Map<number, Map<number, number>>();
    
    // Initialize matrix
    for (let i = 1; i <= maxNumber; i++) {
      matrix.set(i, new Map<number, number>());
    }
    
    // Build transitions from consecutive draws
    for (let i = 0; i < history.length - 1; i++) {
      const currentDraw = history[i].numbers;
      const nextDraw = history[i + 1].numbers;
      
      // Record transitions from each number in current to each in next
      currentDraw.forEach(fromNum => {
        nextDraw.forEach(toNum => {
          const transitions = matrix.get(fromNum)!;
          transitions.set(toNum, (transitions.get(toNum) || 0) + 1);
        });
      });
    }
    
    // Also consider within-draw transitions (pairs)
    history.forEach(draw => {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      for (let i = 0; i < sorted.length - 1; i++) {
        const transitions = matrix.get(sorted[i])!;
        transitions.set(sorted[i + 1], (transitions.get(sorted[i + 1]) || 0) + 1);
      }
    });
    
    return matrix;
  }

  private selectNextNumber(
    currentNumber: number,
    matrix: Map<number, Map<number, number>>,
    exclude: number[],
    minNumber: number,
    maxNumber: number
  ): number | null {
    const transitions = matrix.get(currentNumber);
    if (!transitions || transitions.size === 0) return null;
    
    // Filter out already selected numbers
    const availableTransitions = new Map<number, number>();
    transitions.forEach((weight, num) => {
      if (!exclude.includes(num) && num >= minNumber && num <= maxNumber) {
        availableTransitions.set(num, weight);
      }
    });
    
    if (availableTransitions.size === 0) return null;
    
    // Weighted random selection
    const totalWeight = Array.from(availableTransitions.values()).reduce((a, b) => a + b, 0);
    let random = Math.random() * totalWeight;
    
    for (const [num, weight] of availableTransitions) {
      random -= weight;
      if (random <= 0) return num;
    }
    
    return Array.from(availableTransitions.keys())[0];
  }
}
