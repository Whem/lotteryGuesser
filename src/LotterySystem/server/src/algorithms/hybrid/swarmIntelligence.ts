import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * Swarm Intelligence - Raj Intelligencia
 * =======================================
 * Több "ágens" (hangya/méh) egymástól függetlenül keres számokat,
 * majd a feromonszerű jelzéseik alapján konszenzus alakul ki.
 * 
 * - Hangya kolónia optimalizálás (ACO)
 * - Minden ágens másképp súlyoz
 * - Feromonok erősítik a népszerű útvonalakat
 */
export class SwarmIntelligence extends BaseAlgorithm {
  name = 'swarm_intelligence';
  displayName = 'Swarm Intelligence';
  description = 'Ant colony inspired algorithm where multiple agents vote for optimal numbers';
  category = 'HYBRID' as const;
  complexity = 'ADVANCED' as const;
  isPremium = false;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    if (history.length < 5) {
      return {
        main: this.generateRandomNumbers(config.minNumber, config.maxNumber, config.numbersCount),
        additional: config.hasAdditionalNumbers && config.additionalNumbersCount
          ? this.generateRandomNumbers(config.additionalMinNumber!, config.additionalMaxNumber!, config.additionalNumbersCount)
          : undefined,
        confidence: 0.1, executionTime: performance.now() - startTime,
      };
    }

    // Feromon mátrix
    const pheromones = new Map<number, number>();
    for (let n = config.minNumber; n <= config.maxNumber; n++) {
      pheromones.set(n, 1.0);
    }

    const NUM_AGENTS = 50;
    const NUM_ITERATIONS = 20;
    const EVAPORATION = 0.85;

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      const iterVotes = new Map<number, number>();

      for (let agent = 0; agent < NUM_AGENTS; agent++) {
        // Minden ágens más stratégiát használ
        const strategy = agent % 5;
        const agentPicks = this.agentPick(strategy, config, history, pheromones);
        
        agentPicks.forEach(num => {
          iterVotes.set(num, (iterVotes.get(num) || 0) + 1);
        });
      }

      // Feromon frissítés
      pheromones.forEach((val, num) => {
        const votes = iterVotes.get(num) || 0;
        const newPheromone = val * EVAPORATION + votes * 0.5;
        pheromones.set(num, newPheromone);
      });
    }

    // Végső kiválasztás a feromon szintek alapján
    const sorted = Array.from(pheromones.entries()).sort((a, b) => b[1] - a[1]);
    const mainNumbers = this.balancedSelect(sorted, config);

    // Additional numbers
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      const addPheromones = new Map<number, number>();
      history.forEach((draw, idx) => {
        const w = Math.pow(0.9, idx);
        (draw.additionalNumbers || draw.additional)?.forEach(n => {
          addPheromones.set(n, (addPheromones.get(n) || 0) + w);
        });
      });
      additional = this.selectTopN(addPheromones, config.additionalNumbersCount);
      if (additional.length < config.additionalNumbersCount) {
        additional = this.generateRandomNumbers(
          config.additionalMinNumber!, config.additionalMaxNumber!, config.additionalNumbersCount
        );
      }
    }

    return {
      main: mainNumbers,
      additional,
      confidence: 0.80,
      executionTime: performance.now() - startTime,
    };
  }

  private agentPick(
    strategy: number,
    config: LotteryConfig,
    history: HistoricalDraw[],
    pheromones: Map<number, number>
  ): number[] {
    const scores = new Map<number, number>();

    switch (strategy) {
      case 0: // Frekvencia ágens
        history.forEach((draw, idx) => {
          const w = Math.pow(0.92, idx);
          draw.numbers.forEach(n => scores.set(n, (scores.get(n) || 0) + w * 10));
        });
        break;

      case 1: // Gap ágens
        for (let n = config.minNumber; n <= config.maxNumber; n++) {
          let gap = history.length;
          for (let i = 0; i < history.length; i++) {
            if (history[i].numbers.includes(n)) { gap = i; break; }
          }
          scores.set(n, gap >= 3 && gap <= 8 ? 30 - Math.abs(gap - 5) * 4 : 10);
        }
        break;

      case 2: // Momentum ágens
        history.slice(0, 5).forEach((draw, idx) => {
          draw.numbers.forEach(n => scores.set(n, (scores.get(n) || 0) + (5 - idx) * 8));
        });
        break;

      case 3: // Páros ágens
        const pairs = new Map<string, number>();
        history.forEach(draw => {
          for (let i = 0; i < draw.numbers.length; i++) {
            for (let j = i + 1; j < draw.numbers.length; j++) {
              const key = [draw.numbers[i], draw.numbers[j]].sort((a, b) => a - b).join('-');
              pairs.set(key, (pairs.get(key) || 0) + 1);
            }
          }
        });
        const recent = history[0].numbers;
        for (let n = config.minNumber; n <= config.maxNumber; n++) {
          let pairScore = 0;
          recent.forEach(r => {
            const key = [n, r].sort((a, b) => a - b).join('-');
            pairScore += (pairs.get(key) || 0) * 5;
          });
          scores.set(n, pairScore);
        }
        break;

      case 4: // Random-feromon ágens
        for (let n = config.minNumber; n <= config.maxNumber; n++) {
          scores.set(n, (pheromones.get(n) || 1) * (0.5 + Math.random()));
        }
        break;
    }

    // Feromon hatás minden ágensre
    scores.forEach((score, num) => {
      scores.set(num, score + (pheromones.get(num) || 1) * 0.3);
    });

    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, config.numbersCount)
      .map(([n]) => n);
  }

  private balancedSelect(sorted: [number, number][], config: LotteryConfig): number[] {
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
    return this.fillRemaining(selected, config, config.numbersCount);
  }
}
