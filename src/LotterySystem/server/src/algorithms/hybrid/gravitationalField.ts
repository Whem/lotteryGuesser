import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * Gravitational Field Predictor - Gravitációs Mező
 * ==================================================
 * Fizikai szimulációra épülő algoritmus ahol minden szám "tömege"
 * a frekvenciájával arányos, és gravitációs vonzást fejt ki.
 * 
 * - Számok mint "bolygók" gravitációs mezőben
 * - Vonzás a gyakori számok felé
 * - Taszítás a túl közeli/friss számok között
 * - Lagrange-pontok keresése (egyensúlyi pontok)
 */
export class GravitationalField extends BaseAlgorithm {
  name = 'gravitational_field';
  displayName = 'Gravitational Field';
  description = 'Physics simulation where numbers attract each other based on frequency and patterns';
  category = 'HYBRID' as const;
  complexity = 'ADVANCED' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    if (history.length < 5) {
      return this.randomFallback(config, startTime);
    }

    // 1. Tömeg kiszámítása (frekvencia alapú)
    const mass = new Map<number, number>();
    history.forEach((draw, idx) => {
      const weight = Math.pow(0.93, idx);
      draw.numbers.forEach(n => {
        mass.set(n, (mass.get(n) || 0) + weight);
      });
    });

    // Normalizálás
    const maxMass = Math.max(...Array.from(mass.values()), 1);
    mass.forEach((m, n) => mass.set(n, m / maxMass));

    // 2. Gravitációs potenciál minden számnál
    const potential = new Map<number, number>();

    for (let n = config.minNumber; n <= config.maxNumber; n++) {
      let totalPotential = 0;

      // Vonzás más számok felé (Newton gravitáció: F = m1*m2/r²)
      mass.forEach((m, other) => {
        if (other === n) return;
        const distance = Math.abs(n - other) + 1;
        const force = m / (distance * distance);
        totalPotential += force;
      });

      // Momentum bónusz (utolsó 5 húzás)
      const recentHits = history.slice(0, 5).filter(d => d.numbers.includes(n)).length;
      totalPotential += recentHits * 0.15;

      // Esedékesség bónusz
      let gap = history.length;
      for (let i = 0; i < history.length; i++) {
        if (history[i].numbers.includes(n)) { gap = i; break; }
      }
      if (gap >= 3 && gap <= 8) {
        totalPotential += 0.2 - Math.abs(gap - 5) * 0.03;
      }

      // Taszítás: ha az utolsó húzásban volt, kismértékű taszítás
      if (history[0].numbers.includes(n)) {
        totalPotential *= 0.85;
      }

      potential.set(n, totalPotential);
    }

    // 3. "Lagrange pontok" keresése - számok amelyek egyensúlyban vannak
    // (magas potenciál + jó eloszlás)
    const equilibrium = new Map<number, number>();
    const rangeSize = Math.ceil(config.maxNumber / 5);

    potential.forEach((pot, num) => {
      const range = Math.floor((num - config.minNumber) / rangeSize);
      
      // Ellenőrizzük a szomszédos számokat
      let neighborBalance = 0;
      for (let delta = -3; delta <= 3; delta++) {
        if (delta === 0) continue;
        const neighbor = num + delta;
        if (neighbor >= config.minNumber && neighbor <= config.maxNumber) {
          neighborBalance += potential.get(neighbor) || 0;
        }
      }
      neighborBalance /= 6;

      // Equilibrium score: saját potenciál + szomszédos egyensúly
      equilibrium.set(num, pot * 0.7 + neighborBalance * 0.3);
    });

    // 4. Kiválasztás
    const sorted = Array.from(equilibrium.entries()).sort((a, b) => b[1] - a[1]);
    const selected: number[] = [];
    let oddCount = 0;
    const targetOdd = Math.ceil(config.numbersCount / 2);
    const usedRanges = new Map<number, number>();

    for (const [num] of sorted) {
      if (selected.length >= config.numbersCount) break;
      const isOdd = num % 2 === 1;
      const range = Math.floor((num - config.minNumber) / rangeSize);
      
      if (isOdd && oddCount >= targetOdd) continue;
      if (!isOdd && selected.length - oddCount >= config.numbersCount - targetOdd) continue;
      if ((usedRanges.get(range) || 0) >= 2) continue;
      
      selected.push(num);
      if (isOdd) oddCount++;
      usedRanges.set(range, (usedRanges.get(range) || 0) + 1);
    }

    const mainNumbers = this.fillRemaining(selected, config, config.numbersCount);

    // Additional numbers
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
      confidence: 0.79,
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
