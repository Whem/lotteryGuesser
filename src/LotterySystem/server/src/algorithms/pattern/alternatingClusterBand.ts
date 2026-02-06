import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class AlternatingClusterBand extends BaseAlgorithm {
  name = 'alternating_cluster_band';
  displayName = 'Alternating Cluster Band';
  description = 'Divides numbers into bands and alternates selection from active clusters';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    const range = config.maxNumber - config.minNumber + 1;
    const bandCount = Math.min(config.numbersCount * 2, 10);
    const bandSize = Math.ceil(range / bandCount);
    
    // Analyze band activity in recent draws
    const bandActivity = new Array(bandCount).fill(0).map(() => ({ count: 0, numbers: new Map<number, number>() }));
    
    for (const draw of history.slice(0, 20)) {
      for (const num of draw.numbers) {
        const bandIndex = Math.floor((num - config.minNumber) / bandSize);
        if (bandIndex >= 0 && bandIndex < bandCount) {
          bandActivity[bandIndex].count++;
          bandActivity[bandIndex].numbers.set(num, (bandActivity[bandIndex].numbers.get(num) || 0) + 1);
        }
      }
    }
    
    // Identify alternating pattern
    const lastDraw = history[0]?.numbers || [];
    const lastBands = new Set<number>();
    for (const num of lastDraw) {
      lastBands.add(Math.floor((num - config.minNumber) / bandSize));
    }
    
    // Select from alternating bands (skip last draw's bands)
    const selected: number[] = [];
    const bandOrder = bandActivity
      .map((band, index) => ({ index, activity: band.count, numbers: band.numbers }))
      .filter(b => !lastBands.has(b.index))
      .sort((a, b) => b.activity - a.activity);
    
    // Pick top number from each active band
    for (const band of bandOrder) {
      if (selected.length >= config.numbersCount) break;
      
      const bandNumbers = Array.from(band.numbers.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([num]) => num)
        .filter(n => !selected.includes(n));
      
      if (bandNumbers.length > 0) {
        selected.push(bandNumbers[0]);
      }
    }
    
    // Fill remaining from any band
    while (selected.length < config.numbersCount) {
      const num = config.minNumber + Math.floor(Math.random() * range);
      if (!selected.includes(num)) {
        selected.push(num);
      }
    }
    
    const main = selected.sort((a, b) => a - b).slice(0, config.numbersCount);
    let additional: number[] | undefined;
    
    if (config.hasadditional && config.additionalCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalCount
      );
    }
    
    return {
      main,
      additional,
      confidence: 0.72,
      executionTime: performance.now() - startTime,
    };
  }
}

export const alternatingClusterBand = new AlternatingClusterBand();
