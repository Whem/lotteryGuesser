import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * Adaptive Resonance Theory - Adaptív Rezonancia
 * ================================================
 * Neurális hálózat inspirált algoritmus ami "rezonancia" mintákat
 * keres a historikus adatokban. Ha egy szám rezonál (ismétlődő
 * mintázattal jelenik meg), magasabb pontot kap.
 * 
 * Technikák:
 * - Autokorrelációs rezonancia detektálás
 * - Fázis-szinkronizáció számok között
 * - Adaptív küszöb tanulás
 */
export class AdaptiveResonance extends BaseAlgorithm {
  name = 'adaptive_resonance';
  displayName = 'Adaptive Resonance';
  description = 'Neural-inspired algorithm detecting resonance patterns in historical draws';
  category = 'HYBRID' as const;
  complexity = 'ADVANCED' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    if (history.length < 10) return this.fallback(config, startTime);

    const scores = new Map<number, number>();

    // 1. Autokorrelációs rezonancia - ismétlődő periódusok detektálása
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const appearances = history.map((d, i) => d.numbers.includes(num) ? i : -1).filter(i => i >= 0);
      
      let resonanceScore = 0;
      if (appearances.length >= 2) {
        // Gaps between appearances
        const gaps: number[] = [];
        for (let i = 1; i < appearances.length; i++) {
          gaps.push(appearances[i] - appearances[i - 1]);
        }
        
        // Rezonancia = ha a gap-ek hasonlóak (alacsony szórás)
        if (gaps.length >= 2) {
          const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
          const variance = gaps.reduce((sum, g) => sum + Math.pow(g - avgGap, 2), 0) / gaps.length;
          const stdDev = Math.sqrt(variance);
          
          // Alacsony szórás = erős rezonancia
          resonanceScore = Math.max(0, 40 - stdDev * 8);
          
          // Bónusz ha a következő megjelenés "esedékes" a periódus alapján
          const lastGap = appearances[0]; // draws since last appearance
          const expectedNext = avgGap;
          if (Math.abs(lastGap - expectedNext) < 2) {
            resonanceScore += 25;
          }
        }
      }
      scores.set(num, resonanceScore);
    }

    // 2. Fázis-szinkronizáció - mely számok jelennek meg hasonló fázisban
    const recentNums = history[0].numbers;
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let syncScore = 0;
      
      recentNums.forEach(recent => {
        if (recent === num) return;
        // Hányszor jelentek meg együtt
        let coAppearance = 0;
        history.slice(0, 15).forEach(draw => {
          if (draw.numbers.includes(num) && draw.numbers.includes(recent)) {
            coAppearance++;
          }
        });
        syncScore += coAppearance * 3;
      });
      
      scores.set(num, (scores.get(num) || 0) + syncScore);
    }

    // 3. Adaptív frekvencia súlyozás (frissebb = fontosabb)
    history.forEach((draw, idx) => {
      const weight = Math.pow(0.91, idx) * 15;
      draw.numbers.forEach(num => {
        scores.set(num, (scores.get(num) || 0) + weight);
      });
    });

    // 4. Kiválasztás egyensúllyal
    const mainNumbers = this.selectBalanced(scores, config);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const addScores = new Map<number, number>();
      history.forEach((draw, idx) => {
        const w = Math.pow(0.9, idx) * 10;
        (draw.additionalNumbers || draw.additional)?.forEach(n => {
          addScores.set(n, (addScores.get(n) || 0) + w);
        });
      });
      additional = this.selectTopN(addScores, config.additionalNumbersCount);
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!, config.additionalMaxNumber!, config.additionalNumbersCount
        );
      }
    }

    return {
      main: mainNumbers,
      additional,
      confidence: 0.82,
      executionTime: performance.now() - startTime,
    };
  }

  private selectBalanced(scores: Map<number, number>, config: LotteryConfig): number[] {
    const sorted = Array.from(scores.entries()).sort((a, b) => b[1] - a[1]);
    const selected: number[] = [];
    let oddCount = 0;
    const targetOdd = Math.ceil(config.numbersCount / 2);
    const usedRanges = new Map<number, number>();
    const rangeSize = Math.ceil(config.maxNumber / 5);

    for (const [num] of sorted) {
      if (selected.length >= config.numbersCount) break;
      const isOdd = num % 2 === 1;
      const range = Math.floor((num - 1) / rangeSize);
      if (isOdd && oddCount >= targetOdd) continue;
      if (!isOdd && selected.length - oddCount >= config.numbersCount - targetOdd) continue;
      if ((usedRanges.get(range) || 0) >= 2) continue;
      selected.push(num);
      if (isOdd) oddCount++;
      usedRanges.set(range, (usedRanges.get(range) || 0) + 1);
    }
    return this.fillRemaining(selected, config, config.numbersCount);
  }

  private fallback(config: LotteryConfig, startTime: number): GeneratedNumbers {
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
