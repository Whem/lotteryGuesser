import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class BayesianPrediction extends BaseAlgorithm {
  name = 'bayesian_prediction';
  displayName = 'Bayesian Inference';
  description = 'Applies Bayesian probability to update beliefs about number likelihood';
  category = 'PROBABILITY' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate prior probabilities (uniform)
    const totalNumbers = config.maxNumber - config.minNumber + 1;
    const priorProbability = 1 / totalNumbers;
    
    // Calculate likelihood based on historical frequency
    const frequency = this.calculateFrequency(history);
    const totalDraws = history.length;
    
    // Calculate posterior probabilities
    const posteriors = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const freq = frequency.get(num) || 0;
      const likelihood = totalDraws > 0 ? freq / totalDraws : priorProbability;
      
      // Apply Bayes' theorem: P(num|data) ∝ P(data|num) * P(num)
      // Since we're comparing, we can skip the denominator (same for all)
      const posterior = likelihood * priorProbability;
      
      // Add recency boost (more recent = higher weight)
      const recencyBoost = this.calculateRecencyBoost(num, history);
      
      posteriors.set(num, posterior * (1 + recencyBoost));
    }
    
    // Normalize posteriors
    const totalPosterior = Array.from(posteriors.values()).reduce((a, b) => a + b, 0);
    posteriors.forEach((value, key) => {
      posteriors.set(key, value / totalPosterior);
    });
    
    // Select numbers using posterior-weighted random selection
    const mainNumbers = this.weightedRandomSelect(posteriors, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFreq = this.calculateAdditionalFrequency(history);
      const additionalPosteriors = new Map<number, number>();
      const additionalTotal = config.additionalMaxNumber! - config.additionalMinNumber! + 1;
      
      for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
        const freq = additionalFreq.get(num) || 0;
        const likelihood = totalDraws > 0 ? freq / totalDraws : 1 / additionalTotal;
        additionalPosteriors.set(num, likelihood / additionalTotal);
      }
      
      additional = this.weightedRandomSelect(additionalPosteriors, config.additionalNumbersCount);
      
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
      confidence: 0.73,
      executionTime: performance.now() - startTime,
    };
  }

  private calculateRecencyBoost(num: number, history: HistoricalDraw[]): number {
    const recentDraws = history.slice(0, 20);
    let boost = 0;
    
    recentDraws.forEach((draw, index) => {
      if (draw.numbers.includes(num)) {
        // More recent draws get higher boost
        boost += (20 - index) / 20 * 0.5;
      }
    });
    
    return boost;
  }
}
