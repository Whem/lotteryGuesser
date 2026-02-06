import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class FrequencyAnalysis extends BaseAlgorithm {
  name = 'frequency_analysis';
  displayName = 'Frequency Analysis';
  description = 'Selects numbers based on their historical frequency of appearance';
  category = 'STATISTICAL' as const;
  complexity = 'SIMPLE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();

    // Calculate frequency for main numbers
    const mainFrequency = this.calculateFrequency(history);
    
    // Select top frequent numbers
    const mainNumbers = this.selectTopN(mainFrequency, config.numbersCount);
    
    // Fill if not enough
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Handle additional numbers if needed
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFrequency = this.calculateAdditionalFrequency(history);
      const selectedAdditional = this.selectTopN(additionalFrequency, config.additionalNumbersCount);
      
      additional = selectedAdditional.length >= config.additionalNumbersCount
        ? selectedAdditional
        : this.generateRandomNumbers(
            config.additionalMinNumber!,
            config.additionalMaxNumber!,
            config.additionalNumbersCount
          );
    }

    return {
      main: finalMain,
      additional,
      confidence: Math.min(history.length / 100, 0.9),
      executionTime: performance.now() - startTime,
    };
  }
}
