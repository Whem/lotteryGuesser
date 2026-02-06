import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class HotColdBalance extends BaseAlgorithm {
  name = 'hot_cold_balance';
  displayName = 'Hot & Cold Balance';
  description = 'Combines frequently drawn (hot) and rarely drawn (cold) numbers';
  category = 'STATISTICAL' as const;
  complexity = 'SIMPLE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const frequency = this.calculateFrequency(history);
    const entries = Array.from(frequency.entries()).sort((a, b) => b[1] - a[1]);
    
    // Split into hot and cold
    const hotNumbers = entries.slice(0, Math.ceil(entries.length / 3)).map(e => e[0]);
    const coldNumbers = entries.slice(-Math.ceil(entries.length / 3)).map(e => e[0]);
    
    // Select half from hot, half from cold
    const hotCount = Math.ceil(config.numbersCount / 2);
    const coldCount = config.numbersCount - hotCount;
    
    const selectedHot = this.shuffleAndSelect(hotNumbers, hotCount);
    const selectedCold = this.shuffleAndSelect(coldNumbers, coldCount);
    
    const mainNumbers = [...selectedHot, ...selectedCold].sort((a, b) => a - b);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalFreq = this.calculateAdditionalFrequency(history);
      const additionalEntries = Array.from(additionalFreq.entries()).sort((a, b) => b[1] - a[1]);
      
      const hotAdditional = additionalEntries.slice(0, Math.ceil(additionalEntries.length / 2));
      additional = hotAdditional.slice(0, config.additionalNumbersCount).map(e => e[0]);
      
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
      confidence: 0.65,
      executionTime: performance.now() - startTime,
    };
  }

  private shuffleAndSelect(arr: number[], count: number): number[] {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }
}
