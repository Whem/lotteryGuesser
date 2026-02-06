import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class PatternRecognition extends BaseAlgorithm {
  name = 'pattern_recognition';
  displayName = 'Pattern Recognition';
  description = 'Identifies recurring patterns and sequences in historical draws';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze multiple pattern types
    const deltaPatterns = this.analyzeDeltaPatterns(history);
    const rangePatterns = this.analyzeRangeDistribution(history, config);
    const consecutivePatterns = this.analyzeConsecutiveNumbers(history);
    
    // Combine pattern scores
    const combinedScores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const deltaScore = deltaPatterns.get(num) || 0;
      const rangeScore = rangePatterns.get(num) || 0;
      const consecutiveScore = consecutivePatterns.get(num) || 0;
      
      combinedScores.set(num, deltaScore + rangeScore + consecutiveScore);
    }
    
    // Select top scoring numbers
    const mainNumbers = this.selectTopN(combinedScores, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }

    return {
      main: finalMain,
      additional,
      confidence: 0.68,
      executionTime: performance.now() - startTime,
    };
  }

  private analyzeDeltaPatterns(history: HistoricalDraw[]): Map<number, number> {
    // Analyze differences between consecutive numbers
    const deltaScores = new Map<number, number>();
    const commonDeltas = new Map<number, number>();
    
    // Find common deltas
    history.forEach(draw => {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      for (let i = 1; i < sorted.length; i++) {
        const delta = sorted[i] - sorted[i - 1];
        commonDeltas.set(delta, (commonDeltas.get(delta) || 0) + 1);
      }
    });
    
    // Get most common deltas
    const topDeltas = Array.from(commonDeltas.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(e => e[0]);
    
    // Score numbers based on delta relationships with recent draws
    const lastDraw = history[0]?.numbers || [];
    
    lastDraw.forEach(num => {
      topDeltas.forEach(delta => {
        const nextNum = num + delta;
        const prevNum = num - delta;
        
        if (nextNum > 0) deltaScores.set(nextNum, (deltaScores.get(nextNum) || 0) + 1);
        if (prevNum > 0) deltaScores.set(prevNum, (deltaScores.get(prevNum) || 0) + 1);
      });
    });
    
    return deltaScores;
  }

  private analyzeRangeDistribution(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const rangeSize = Math.ceil((config.maxNumber - config.minNumber + 1) / config.numbersCount);
    const rangeDistribution = new Map<number, number>();
    
    // Analyze which ranges typically have numbers
    history.forEach(draw => {
      draw.numbers.forEach(num => {
        const range = Math.floor((num - config.minNumber) / rangeSize);
        rangeDistribution.set(range, (rangeDistribution.get(range) || 0) + 1);
      });
    });
    
    // Score numbers based on their range's historical frequency
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const range = Math.floor((num - config.minNumber) / rangeSize);
      const rangeFreq = rangeDistribution.get(range) || 0;
      scores.set(num, rangeFreq);
    }
    
    return scores;
  }

  private analyzeConsecutiveNumbers(history: HistoricalDraw[]): Map<number, number> {
    const consecutiveScores = new Map<number, number>();
    
    // Count how often consecutive pairs appear
    history.forEach(draw => {
      const sorted = [...draw.numbers].sort((a, b) => a - b);
      
      for (let i = 0; i < sorted.length - 1; i++) {
        if (sorted[i + 1] - sorted[i] === 1) {
          // Consecutive pair found
          consecutiveScores.set(sorted[i], (consecutiveScores.get(sorted[i]) || 0) + 1);
          consecutiveScores.set(sorted[i + 1], (consecutiveScores.get(sorted[i + 1]) || 0) + 1);
        }
      }
    });
    
    return consecutiveScores;
  }
}
