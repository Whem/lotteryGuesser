import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class EnsembleVoting extends BaseAlgorithm {
  name = 'ensemble_voting';
  displayName = 'Ensemble Voting';
  description = 'Combines multiple strategies through democratic voting';
  category = 'ENSEMBLE' as const;
  complexity = 'COMPLEX' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Collect votes from different strategies
    const votes = new Map<number, number>();
    
    // Strategy 1: Frequency-based
    const frequencySelection = this.frequencyStrategy(history, config);
    frequencySelection.forEach(num => votes.set(num, (votes.get(num) || 0) + 1));
    
    // Strategy 2: Gap-based
    const gapSelection = this.gapStrategy(history, config);
    gapSelection.forEach(num => votes.set(num, (votes.get(num) || 0) + 1));
    
    // Strategy 3: Recent trends
    const trendSelection = this.trendStrategy(history, config);
    trendSelection.forEach(num => votes.set(num, (votes.get(num) || 0) + 1));
    
    // Strategy 4: Range distribution
    const rangeSelection = this.rangeStrategy(history, config);
    rangeSelection.forEach(num => votes.set(num, (votes.get(num) || 0) + 1));
    
    // Strategy 5: Odd/Even balance
    const balanceSelection = this.balanceStrategy(history, config);
    balanceSelection.forEach(num => votes.set(num, (votes.get(num) || 0) + 1));
    
    // Select numbers with most votes
    const mainNumbers = this.selectTopN(votes, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalVotes = new Map<number, number>();
      
      // Simple frequency voting for additional
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
      confidence: 0.72,
      executionTime: performance.now() - startTime,
    };
  }

  private frequencyStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const frequency = this.calculateFrequency(history);
    return this.selectTopN(frequency, config.numbersCount * 2);
  }

  private gapStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const gaps = this.calculateGaps(history, config.maxNumber);
    return this.selectTopN(gaps, config.numbersCount * 2);
  }

  private trendStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const recentFreq = new Map<number, number>();
    const recent = history.slice(0, 10);
    
    recent.forEach((draw, index) => {
      const weight = 10 - index;
      draw.numbers.forEach(num => {
        recentFreq.set(num, (recentFreq.get(num) || 0) + weight);
      });
    });
    
    return this.selectTopN(recentFreq, config.numbersCount * 2);
  }

  private rangeStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    const rangeSize = Math.ceil((config.maxNumber - config.minNumber + 1) / config.numbersCount);
    const selected: number[] = [];
    
    for (let i = 0; i < config.numbersCount; i++) {
      const rangeStart = config.minNumber + i * rangeSize;
      const rangeEnd = Math.min(rangeStart + rangeSize - 1, config.maxNumber);
      
      // Find most frequent in this range
      const rangeFreq = new Map<number, number>();
      history.forEach(draw => {
        draw.numbers.forEach(num => {
          if (num >= rangeStart && num <= rangeEnd) {
            rangeFreq.set(num, (rangeFreq.get(num) || 0) + 1);
          }
        });
      });
      
      const topInRange = this.selectTopN(rangeFreq, 2);
      selected.push(...topInRange);
    }
    
    return selected;
  }

  private balanceStrategy(history: HistoricalDraw[], config: LotteryConfig): number[] {
    // Analyze typical odd/even ratio
    const ratios = history.map(draw => {
      const oddCount = draw.numbers.filter(n => n % 2 === 1).length;
      return oddCount / draw.numbers.length;
    });
    
    const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    const targetOdd = Math.round(config.numbersCount * avgRatio);
    
    const oddNumbers: number[] = [];
    const evenNumbers: number[] = [];
    
    for (let i = config.minNumber; i <= config.maxNumber; i++) {
      if (i % 2 === 1) oddNumbers.push(i);
      else evenNumbers.push(i);
    }
    
    // Get frequent odds and evens
    const frequency = this.calculateFrequency(history);
    
    const sortedOdds = oddNumbers
      .map(n => ({ num: n, freq: frequency.get(n) || 0 }))
      .sort((a, b) => b.freq - a.freq)
      .slice(0, targetOdd * 2)
      .map(x => x.num);
    
    const sortedEvens = evenNumbers
      .map(n => ({ num: n, freq: frequency.get(n) || 0 }))
      .sort((a, b) => b.freq - a.freq)
      .slice(0, (config.numbersCount - targetOdd) * 2)
      .map(x => x.num);
    
    return [...sortedOdds, ...sortedEvens];
  }
}
