import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class WeightedEnsemble extends BaseAlgorithm {
  name = 'weighted_ensemble';
  displayName = 'Weighted Ensemble';
  description = 'Combines multiple strategies with performance-based weights';
  category = 'ENSEMBLE' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Calculate strategy weights based on backtest performance
    const strategyPerformance = this.backtestStrategies(history, config);
    
    // Collect weighted votes
    const weightedScores = new Map<number, number>();
    
    // Apply each strategy with its weight
    strategyPerformance.forEach(({ strategy, weight }) => {
      const selection = strategy(history.slice(10), config);
      selection.forEach(num => {
        weightedScores.set(num, (weightedScores.get(num) || 0) + weight);
      });
    });
    
    // Select top weighted numbers
    const mainNumbers = this.selectTopN(weightedScores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
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
      confidence: 0.8,
      executionTime: performance.now() - startTime,
    };
  }

  private backtestStrategies(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): { strategy: (h: HistoricalDraw[], c: LotteryConfig) => number[]; weight: number }[] {
    const strategies = [
      { name: 'frequency', fn: this.frequencyStrategy.bind(this) },
      { name: 'gap', fn: this.gapStrategy.bind(this) },
      { name: 'weighted', fn: this.weightedStrategy.bind(this) },
      { name: 'pattern', fn: this.patternStrategy.bind(this) },
    ];
    
    // Backtest on last 10 draws
    const testDraws = history.slice(0, 10);
    const trainingHistory = history.slice(10);
    
    const scores = strategies.map(({ name, fn }) => {
      let totalScore = 0;
      
      testDraws.forEach(testDraw => {
        const prediction = fn(trainingHistory, config);
        const matches = prediction.filter(n => testDraw.numbers.includes(n)).length;
        totalScore += matches;
      });
      
      return { strategy: fn, score: totalScore, name };
    });
    
    // Normalize scores to weights
    const totalScore = scores.reduce((sum, s) => sum + s.score, 0) || 1;
    
    return scores.map(s => ({
      strategy: s.strategy,
      weight: (s.score / totalScore) + 0.1, // Minimum weight of 0.1
    }));
  }

  private frequencyStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const frequency = this.calculateFrequency(history);
    return this.selectTopN(frequency, config.numbersCount * 2);
  }

  private gapStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const gaps = this.calculateGaps(history, config.maxNumber);
    return this.selectTopN(gaps, config.numbersCount * 2);
  }

  private weightedStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const weights = new Map<number, number>();
    
    history.forEach((draw, index) => {
      const decay = Math.exp(-index / 30);
      draw.numbers.forEach(num => {
        weights.set(num, (weights.get(num) || 0) + decay);
      });
    });
    
    return this.selectTopN(weights, config.numbersCount * 2);
  }

  private patternStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    // Look for repeating patterns in last draw
    const lastDraw = history[0]?.numbers || [];
    const scores = new Map<number, number>();
    
    // Consecutive number patterns
    lastDraw.forEach(num => {
      if (num + 1 <= config.maxNumber) scores.set(num + 1, (scores.get(num + 1) || 0) + 2);
      if (num - 1 >= config.minNumber) scores.set(num - 1, (scores.get(num - 1) || 0) + 2);
      if (num + 10 <= config.maxNumber) scores.set(num + 10, (scores.get(num + 10) || 0) + 1);
      if (num - 10 >= config.minNumber) scores.set(num - 10, (scores.get(num - 10) || 0) + 1);
    });
    
    // Add frequency bonus
    const frequency = this.calculateFrequency(history);
    frequency.forEach((freq, num) => {
      scores.set(num, (scores.get(num) || 0) + freq / history.length);
    });
    
    return this.selectTopN(scores, config.numbersCount * 2);
  }
}
