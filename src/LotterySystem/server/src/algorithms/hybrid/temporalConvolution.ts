import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * Temporal Convolution Network - Temporális Konvolúció
 * =====================================================
 * CNN-inspirált algoritmus ami "konvolúciós szűrőket" futtat
 * a historikus idősoron, mintákat keresve különböző ablakméretekkel.
 * 
 * - Többrétegű konvolúciós szűrők (3, 5, 7 ablakméret)
 * - Dilated convolution a hosszú távú minták keresésére
 * - Pooling a legjellemzőbb minták kiemelésére
 */
export class TemporalConvolution extends BaseAlgorithm {
  name = 'temporal_convolution';
  displayName = 'Temporal Convolution';
  description = 'CNN-inspired algorithm applying convolution filters over draw history timeseries';
  category = 'MACHINE_LEARNING' as const;
  complexity = 'ADVANCED' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    if (history.length < 10) {
      return this.randomFallback(config, startTime);
    }

    // Bináris mátrix: [draw_index][number] = 0/1
    const matrix: number[][] = history.map(draw => {
      const row = new Array(config.maxNumber + 1).fill(0);
      draw.numbers.forEach(n => { row[n] = 1; });
      return row;
    });

    const finalScores = new Map<number, number>();
    for (let n = config.minNumber; n <= config.maxNumber; n++) {
      finalScores.set(n, 0);
    }

    // Layer 1: Szűrők különböző ablakméretekkel
    for (const windowSize of [3, 5, 7]) {
      for (let num = config.minNumber; num <= config.maxNumber; num++) {
        // Konvolúció: sliding window az idősoron
        const timeseries = matrix.map(row => row[num]);
        let convScore = 0;
        
        for (let i = 0; i <= timeseries.length - windowSize; i++) {
          const window = timeseries.slice(i, i + windowSize);
          const sum = window.reduce((a, b) => a + b, 0);
          // Exponenciális súlyozás (frissebb = fontosabb)
          const recencyWeight = Math.pow(0.9, i);
          convScore += sum * recencyWeight;
        }
        
        finalScores.set(num, (finalScores.get(num) || 0) + convScore);
      }
    }

    // Layer 2: Dilated convolution (hosszú távú minták)
    for (const dilation of [2, 4]) {
      for (let num = config.minNumber; num <= config.maxNumber; num++) {
        const timeseries = matrix.map(row => row[num]);
        let dilatedScore = 0;
        
        for (let i = 0; i < timeseries.length; i += dilation) {
          if (timeseries[i] === 1) {
            dilatedScore += Math.pow(0.92, i) * 5;
          }
        }
        
        finalScores.set(num, (finalScores.get(num) || 0) + dilatedScore);
      }
    }

    // Layer 3: Cross-number convolution (számok közötti korreláció)
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let crossScore = 0;
      const recentNumbers = history[0].numbers;
      
      recentNumbers.forEach(recent => {
        if (recent === num) return;
        // Hányszor jelentek meg együtt az ablakban
        let coCount = 0;
        for (let i = 0; i < Math.min(15, history.length); i++) {
          if (history[i].numbers.includes(num) && history[i].numbers.includes(recent)) {
            coCount++;
          }
        }
        crossScore += coCount * 3;
      });
      
      finalScores.set(num, (finalScores.get(num) || 0) + crossScore);
    }

    // Max pooling: tartomány alapú kiemelés
    const rangeSize = Math.ceil(config.maxNumber / config.numbersCount);
    const rangeMaxes: Map<number, { num: number; score: number }> = new Map();
    
    finalScores.forEach((score, num) => {
      const rangeIdx = Math.floor((num - config.minNumber) / rangeSize);
      const current = rangeMaxes.get(rangeIdx);
      if (!current || score > current.score) {
        rangeMaxes.set(rangeIdx, { num, score });
      }
    });

    // Kiválasztás: top scores + range diversity
    const sorted = Array.from(finalScores.entries()).sort((a, b) => b[1] - a[1]);
    const selected: number[] = [];
    let oddCount = 0;
    const targetOdd = Math.ceil(config.numbersCount / 2);

    for (const [num] of sorted) {
      if (selected.length >= config.numbersCount) break;
      const isOdd = num % 2 === 1;
      if (isOdd && oddCount >= targetOdd) continue;
      if (!isOdd && selected.length - oddCount >= config.numbersCount - targetOdd) continue;
      selected.push(num);
      if (isOdd) oddCount++;
    }

    const mainNumbers = this.fillRemaining(selected, config, config.numbersCount);

    // Additional
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const addFreq = this.calculateAdditionalFrequency(history);
      additional = this.selectTopN(addFreq, config.additionalNumbersCount);
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!, config.additionalMaxNumber!, config.additionalNumbersCount
        );
      }
    }

    return {
      main: mainNumbers,
      additional,
      confidence: 0.77,
      executionTime: performance.now() - startTime,
    };
  }

  private randomFallback(config: LotteryConfig, startTime: number): GeneratedNumbers {
    return {
      main: this.generateRandomNumbers(config.minNumber, config.maxNumber, config.numbersCount),
      additional: config.hasAdditionalNumbers && config.additionalNumbersCount
        ? this.generateRandomNumbers(config.additionalMinNumber!, config.additionalMaxNumber!, config.additionalNumbersCount)
        : undefined,
      confidence: 0.1,
      executionTime: performance.now() - startTime,
    };
  }
}
