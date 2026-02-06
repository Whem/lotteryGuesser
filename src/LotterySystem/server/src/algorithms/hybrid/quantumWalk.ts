import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * Quantum Walk Predictor - Kvantum Séta
 * ======================================
 * Kvantumszámítás-inspirált algoritmus ami "szuperpozíció"-ban
 * tartja a számokat és valószínűségi hullámfüggvénnyel választ.
 * 
 * - Valószínűségi amplitúdó minden számhoz
 * - Interferencia minták (konstruktív/destruktív)
 * - "Mérés" = végleges kiválasztás
 */
export class QuantumWalk extends BaseAlgorithm {
  name = 'quantum_walk';
  displayName = 'Quantum Walk';
  description = 'Quantum computing inspired algorithm using superposition and interference patterns';
  category = 'HYBRID' as const;
  complexity = 'ADVANCED' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    if (history.length < 5) return this.randomFallback(config, startTime);

    // Amplitúdók inicializálása (komplex számok egyszerűsítve: [real, imaginary])
    const amplitudes = new Map<number, [number, number]>();
    for (let n = config.minNumber; n <= config.maxNumber; n++) {
      amplitudes.set(n, [0.5, 0.5]); // Egyenlő szuperpozíció
    }

    // 1. Frekvencia operátor - módosítja az amplitúdókat
    const freq = this.calculateFrequency(history);
    const maxFreq = Math.max(...Array.from(freq.values()), 1);
    
    freq.forEach((count, num) => {
      const [re, im] = amplitudes.get(num) || [0.5, 0.5];
      const boost = (count / maxFreq) * 0.6;
      amplitudes.set(num, [re + boost, im + boost * 0.5]);
    });

    // 2. Gap interferencia - konstruktív ha "esedékes"
    const gaps = this.calculateGaps(history, config.maxNumber);
    gaps.forEach((gap, num) => {
      const [re, im] = amplitudes.get(num) || [0.5, 0.5];
      // Konstruktív interferencia gap 3-8 között
      if (gap >= 3 && gap <= 8) {
        const constructive = 0.4 - Math.abs(gap - 5) * 0.06;
        amplitudes.set(num, [re + constructive, im + constructive]);
      } else if (gap > 12) {
        // Destruktív interferencia - túl rég volt
        amplitudes.set(num, [re * 0.7, im * 0.7]);
      }
    });

    // 3. Momentum hullám
    const recentFreq = new Map<number, number>();
    history.slice(0, 5).forEach((draw, idx) => {
      const w = 5 - idx;
      draw.numbers.forEach(n => recentFreq.set(n, (recentFreq.get(n) || 0) + w));
    });
    
    recentFreq.forEach((score, num) => {
      const [re, im] = amplitudes.get(num) || [0.5, 0.5];
      amplitudes.set(num, [re + score * 0.04, im + score * 0.02]);
    });

    // 4. Szomszédsági interferencia
    history.slice(0, 10).forEach(draw => {
      draw.numbers.forEach(num => {
        // Szomszédos számok kismértékű boost-ot kapnak
        for (const delta of [-2, -1, 1, 2]) {
          const neighbor = num + delta;
          if (neighbor >= config.minNumber && neighbor <= config.maxNumber) {
            const [re, im] = amplitudes.get(neighbor) || [0.5, 0.5];
            amplitudes.set(neighbor, [re + 0.03, im + 0.01]);
          }
        }
      });
    });

    // 5. "Mérés" - valószínűség = |amplitúdó|²
    const probabilities = new Map<number, number>();
    amplitudes.forEach(([re, im], num) => {
      probabilities.set(num, re * re + im * im);
    });

    // Kiválasztás tartomány egyensúllyal
    const sorted = Array.from(probabilities.entries()).sort((a, b) => b[1] - a[1]);
    const mainNumbers = this.balancedSelect(sorted, config);

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
      confidence: 0.78,
      executionTime: performance.now() - startTime,
    };
  }

  private balancedSelect(sorted: [number, number][], config: LotteryConfig): number[] {
    const selected: number[] = [];
    let oddCount = 0;
    const targetOdd = Math.ceil(config.numbersCount / 2);
    const rangeSize = Math.ceil(config.maxNumber / config.numbersCount);

    for (const [num] of sorted) {
      if (selected.length >= config.numbersCount) break;
      const isOdd = num % 2 === 1;
      if (isOdd && oddCount >= targetOdd) continue;
      if (!isOdd && selected.length - oddCount >= config.numbersCount - targetOdd) continue;
      selected.push(num);
      if (isOdd) oddCount++;
    }
    return this.fillRemaining(selected, config, config.numbersCount);
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
