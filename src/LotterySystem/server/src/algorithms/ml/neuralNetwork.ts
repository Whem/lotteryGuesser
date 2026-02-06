import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class NeuralNetwork extends BaseAlgorithm {
  name = 'neural_network';
  displayName = 'Neural Network';
  description = 'Simple neural network simulation for number prediction';
  category = 'MACHINE_LEARNING' as const;
  complexity = 'ADVANCED' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Simplified neural network using weighted inputs
    const inputWeights = this.trainWeights(history, config);
    const activations = this.forwardPass(inputWeights, history, config);
    
    // Select numbers based on activation scores
    const mainNumbers = this.selectTopN(activations, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalActivations = this.trainAdditionalWeights(history, config);
      additional = this.selectTopN(additionalActivations, config.additionalNumbersCount);
      
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
      confidence: 0.78,
      executionTime: performance.now() - startTime,
    };
  }

  private trainWeights(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const weights = new Map<number, number>();
    
    // Initialize weights
    for (let i = config.minNumber; i <= config.maxNumber; i++) {
      weights.set(i, 0.5);
    }
    
    // Train weights based on history (simple backprop simulation)
    const learningRate = 0.1;
    const recentHistory = history.slice(0, 100);
    
    recentHistory.forEach((draw, drawIndex) => {
      const timeDecay = Math.exp(-drawIndex / 50);
      
      draw.numbers.forEach(num => {
        const currentWeight = weights.get(num) || 0.5;
        // Increase weight for numbers that appeared
        const newWeight = currentWeight + (learningRate * timeDecay * (1 - currentWeight));
        weights.set(num, Math.min(1, newWeight));
      });
      
      // Slightly decrease weights for numbers that didn't appear
      for (let i = config.minNumber; i <= config.maxNumber; i++) {
        if (!draw.numbers.includes(i)) {
          const currentWeight = weights.get(i) || 0.5;
          const newWeight = currentWeight - (learningRate * 0.1 * timeDecay * currentWeight);
          weights.set(i, Math.max(0, newWeight));
        }
      }
    });
    
    return weights;
  }

  private forwardPass(
    weights: Map<number, number>,
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const activations = new Map<number, number>();
    
    // Add contextual features
    const frequency = this.calculateFrequency(history);
    const gaps = this.calculateGaps(history, config.maxNumber);
    
    // Combine features through activation function
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const weight = weights.get(num) || 0.5;
      const freq = (frequency.get(num) || 0) / history.length;
      const gap = (gaps.get(num) || history.length) / history.length;
      
      // Sigmoid-like activation combining features
      const input = weight * 0.4 + freq * 0.3 + gap * 0.3;
      const activation = 1 / (1 + Math.exp(-10 * (input - 0.5)));
      
      activations.set(num, activation);
    }
    
    return activations;
  }

  private trainAdditionalWeights(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const weights = new Map<number, number>();
    
    for (let i = config.additionalMinNumber!; i <= config.additionalMaxNumber!; i++) {
      weights.set(i, 0.5);
    }
    
    history.slice(0, 100).forEach((draw, index) => {
      const timeDecay = Math.exp(-index / 50);
      draw.additionalNumbers?.forEach(num => {
        const current = weights.get(num) || 0.5;
        weights.set(num, Math.min(1, current + 0.1 * timeDecay));
      });
    });
    
    return weights;
  }
}
