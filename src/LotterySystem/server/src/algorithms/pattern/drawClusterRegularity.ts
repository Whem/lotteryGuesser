import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class DrawClusterRegularity extends BaseAlgorithm {
  name = 'draw_cluster_regularity';
  displayName = 'Draw Cluster Regularity';
  description = 'Detects regular cluster patterns across consecutive draws';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'MODERATE' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Analyze cluster regularity patterns
    const clusterSize = Math.ceil((config.maxNumber - config.minNumber + 1) / 5);
    const clusterActivity: number[][] = [];
    
    // Track which clusters are active in each draw
    for (const draw of history.slice(0, 20)) {
      const clusters = new Array(5).fill(0);
      for (const num of draw.numbers) {
        const clusterIndex = Math.floor((num - config.minNumber) / clusterSize);
        if (clusterIndex >= 0 && clusterIndex < 5) {
          clusters[clusterIndex]++;
        }
      }
      clusterActivity.push(clusters);
    }
    
    // Find regular patterns (clusters that appear consistently)
    const clusterScores = new Array(5).fill(0);
    for (let i = 0; i < clusterActivity.length; i++) {
      for (let j = 0; j < 5; j++) {
        if (clusterActivity[i][j] > 0) {
          clusterScores[j] += 1 / (i + 1); // Weight recent more
        }
      }
    }
    
    // Select from top scoring clusters
    const frequency = this.calculateFrequency(history);
    const selected: number[] = [];
    
    const sortedClusters = clusterScores
      .map((score, index) => ({ index, score }))
      .sort((a, b) => b.score - a.score);
    
    for (const cluster of sortedClusters) {
      if (selected.length >= config.numbersCount) break;
      
      const clusterStart = config.minNumber + cluster.index * clusterSize;
      const clusterEnd = Math.min(clusterStart + clusterSize - 1, config.maxNumber);
      
      // Get numbers in this cluster sorted by frequency
      const clusterNums: [number, number][] = [];
      for (let num = clusterStart; num <= clusterEnd; num++) {
        clusterNums.push([num, frequency.get(num) || 0]);
      }
      clusterNums.sort((a, b) => b[1] - a[1]);
      
      // Add top number from cluster
      for (const [num] of clusterNums) {
        if (!selected.includes(num)) {
          selected.push(num);
          break;
        }
      }
    }
    
    // Fill remaining
    while (selected.length < config.numbersCount) {
      const num = config.minNumber + Math.floor(Math.random() * (config.maxNumber - config.minNumber + 1));
      if (!selected.includes(num)) {
        selected.push(num);
      }
    }
    
    const main = selected.sort((a, b) => a - b).slice(0, config.numbersCount);
    let additional: number[] | undefined;
    
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }
    
    return { main, additional, confidence: 0.67, executionTime: performance.now() - startTime };
  }
}

export const drawClusterRegularity = new DrawClusterRegularity();
