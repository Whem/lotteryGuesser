import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class ClusterAnalysis extends BaseAlgorithm {
  name = 'cluster_analysis';
  displayName = 'Cluster Analysis';
  description = 'Groups numbers into clusters and selects from active clusters';
  category = 'PATTERN_RECOGNITION' as const;
  complexity = 'MODERATE' as const;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Define clusters (ranges of numbers)
    const clusterSize = Math.ceil((config.maxNumber - config.minNumber + 1) / 5);
    const clusters: number[][] = [];
    
    for (let i = 0; i < 5; i++) {
      const start = config.minNumber + i * clusterSize;
      const end = Math.min(start + clusterSize - 1, config.maxNumber);
      const cluster: number[] = [];
      for (let num = start; num <= end; num++) {
        cluster.push(num);
      }
      clusters.push(cluster);
    }
    
    // Analyze cluster activity in recent draws
    const recentHistory = history.slice(0, 50);
    const clusterActivity = clusters.map((cluster, index) => {
      let activity = 0;
      recentHistory.forEach((draw, drawIndex) => {
        const weight = 1 - (drawIndex / recentHistory.length); // Recent draws weighted more
        const numbersInCluster = draw.numbers.filter(n => cluster.includes(n)).length;
        activity += numbersInCluster * weight;
      });
      return { index, activity, cluster };
    });
    
    // Sort clusters by activity
    clusterActivity.sort((a, b) => b.activity - a.activity);
    
    // Select numbers from top clusters
    const mainNumbers: number[] = [];
    const numbersPerCluster = Math.ceil(config.numbersCount / 3);
    
    for (let i = 0; i < 3 && mainNumbers.length < config.numbersCount; i++) {
      const activeCluster = clusterActivity[i].cluster;
      
      // Get frequency within this cluster
      const clusterFreq = new Map<number, number>();
      activeCluster.forEach(num => clusterFreq.set(num, 0));
      
      history.forEach(draw => {
        draw.numbers.forEach(num => {
          if (clusterFreq.has(num)) {
            clusterFreq.set(num, clusterFreq.get(num)! + 1);
          }
        });
      });
      
      // Select top numbers from cluster
      const topFromCluster = this.selectTopN(clusterFreq, numbersPerCluster);
      topFromCluster.forEach(num => {
        if (mainNumbers.length < config.numbersCount && !mainNumbers.includes(num)) {
          mainNumbers.push(num);
        }
      });
    }
    
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount)
      .sort((a, b) => a - b);

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
      confidence: 0.65,
      executionTime: performance.now() - startTime,
    };
  }
}
