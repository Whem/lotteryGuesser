import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

export class RandomForest extends BaseAlgorithm {
  name = 'random_forest';
  displayName = 'Random Forest';
  description = 'Ensemble of decision trees for robust predictions';
  category = 'MACHINE_LEARNING' as const;
  complexity = 'ADVANCED' as const;
  isPremium = true;

  private readonly treeCount = 10;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Build multiple decision trees
    const treeVotes = new Map<number, number>();
    
    for (let i = 0; i < this.treeCount; i++) {
      const treeSelection = this.buildTree(config, history, i);
      treeSelection.forEach(num => {
        treeVotes.set(num, (treeVotes.get(num) || 0) + 1);
      });
    }
    
    // Select numbers with most votes
    const mainNumbers = this.selectTopN(treeVotes, config.numbersCount);
    const finalMain = this.fillRemaining(mainNumbers, config, config.numbersCount);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const additionalVotes = new Map<number, number>();
      
      for (let i = 0; i < this.treeCount; i++) {
        const selection = this.buildAdditionalTree(config, history, i);
        selection.forEach(num => {
          additionalVotes.set(num, (additionalVotes.get(num) || 0) + 1);
        });
      }
      
      additional = this.selectTopN(additionalVotes, config.additionalNumbersCount);
      
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
      confidence: 0.76,
      executionTime: performance.now() - startTime,
    };
  }

  private buildTree(
    config: LotteryConfig,
    history: HistoricalDraw[],
    treeIndex: number
  ): number[] {
    // Bootstrap sample
    const sampleSize = Math.floor(history.length * 0.7);
    const sample = this.bootstrapSample(history, sampleSize, treeIndex);
    
    // Feature selection (different features for each tree)
    const featureSet = this.selectFeatures(treeIndex);
    
    // Calculate scores based on selected features
    const scores = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let score = 0;
      
      if (featureSet.includes('frequency')) {
        const freq = sample.filter(d => d.numbers.includes(num)).length;
        score += freq / sample.length;
      }
      
      if (featureSet.includes('recency')) {
        const lastAppearance = sample.findIndex(d => d.numbers.includes(num));
        score += lastAppearance === -1 ? 0 : 1 - (lastAppearance / sample.length);
      }
      
      if (featureSet.includes('gap')) {
        let gapSum = 0;
        let gapCount = 0;
        let lastSeen = -1;
        
        sample.forEach((d, idx) => {
          if (d.numbers.includes(num)) {
            if (lastSeen >= 0) {
              gapSum += idx - lastSeen;
              gapCount++;
            }
            lastSeen = idx;
          }
        });
        
        const avgGap = gapCount > 0 ? gapSum / gapCount : sample.length;
        const currentGap = lastSeen >= 0 ? 0 : sample.length;
        score += currentGap / (avgGap + 1);
      }
      
      scores.set(num, score);
    }
    
    return this.selectTopN(scores, config.numbersCount);
  }

  private buildAdditionalTree(
    config: LotteryConfig,
    history: HistoricalDraw[],
    treeIndex: number
  ): number[] {
    const sample = this.bootstrapSample(history, Math.floor(history.length * 0.7), treeIndex);
    const scores = new Map<number, number>();
    
    for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
      const freq = sample.filter(d => d.additionalNumbers?.includes(num)).length;
      scores.set(num, freq / sample.length);
    }
    
    return this.selectTopN(scores, config.additionalNumbersCount!);
  }

  private bootstrapSample(
    history: HistoricalDraw[],
    size: number,
    seed: number
  ): HistoricalDraw[] {
    const sample: HistoricalDraw[] = [];
    const random = this.seededRandom(seed);
    
    for (let i = 0; i < size; i++) {
      const index = Math.floor(random() * history.length);
      sample.push(history[index]);
    }
    
    return sample;
  }

  private selectFeatures(treeIndex: number): string[] {
    const allFeatures = ['frequency', 'recency', 'gap'];
    const featureCount = 2 + (treeIndex % 2); // 2 or 3 features
    
    // Deterministic feature selection based on tree index
    const selected: string[] = [];
    for (let i = 0; i < featureCount; i++) {
      selected.push(allFeatures[(treeIndex + i) % allFeatures.length]);
    }
    
    return selected;
  }

  private seededRandom(seed: number): () => number {
    let state = seed;
    return () => {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      return state / 0x7fffffff;
    };
  }
}
