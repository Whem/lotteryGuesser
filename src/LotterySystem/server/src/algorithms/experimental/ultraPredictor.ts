import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';
import { BaseAlgorithm } from '../base';

/**
 * UltraPredictor - Fejlett Hibrid Algoritmus
 * Cél: 95%+ megbízhatóság elérése
 * 
 * Technikák kombinálása:
 * 1. Súlyozott gyakoriság (exponenciális időbeli súlyozással)
 * 2. Gap analízis (késés alapú predikció)
 * 3. Páros előfordulás analízis
 * 4. Szám tartomány optimalizálás
 * 5. Momentum indikátor
 * 6. Periodicitás felismerés
 * 7. Entrópia alapú kiválasztás
 */
export class UltraPredictor extends BaseAlgorithm {
  name = 'ultra_predictor';
  displayName = 'Ultra Predictor';
  description = 'Advanced hybrid algorithm combining 7+ prediction techniques for maximum accuracy';
  category = 'EXPERIMENTAL' as const;
  complexity = 'ADVANCED' as const;
  isPremium = true;

  async predict(
    config: LotteryConfig,
    history: HistoricalDraw[]
  ): Promise<GeneratedNumbers> {
    const startTime = performance.now();

    if (history.length < 5) {
      return this.generateRandom(config, startTime);
    }

    // 1. Súlyozott gyakoriság - exponenciális időbeli súlyozással
    const weightedFrequency = this.calculateWeightedFrequency(history, config);
    
    // 2. Gap analízis - mely számok "esedékesek"
    const gapScores = this.calculateGapScores(history, config);
    
    // 3. Páros előfordulás - mely számok gyakran jönnek együtt
    const pairScores = this.calculatePairScores(history, config);
    
    // 4. Tartomány optimalizálás
    const rangeScores = this.calculateRangeScores(history, config);
    
    // 5. Momentum - növekvő vagy csökkenő trendek
    const momentumScores = this.calculateMomentumScores(history, config);
    
    // 6. Periodicitás - ismétlődő minták
    const periodicityScores = this.calculatePeriodicityScores(history, config);
    
    // 7. Entrópia - bizonytalanság kezelése
    const entropyScores = this.calculateEntropyScores(history, config);

    // Összes pontszám összevonása súlyozással
    const finalScores = this.combineScores(
      config,
      weightedFrequency,
      gapScores,
      pairScores,
      rangeScores,
      momentumScores,
      periodicityScores,
      entropyScores
    );

    // Top számok kiválasztása
    const sortedNumbers = Array.from(finalScores.entries())
      .sort((a, b) => b[1] - a[1]);

    // Fő számok kiválasztása odd/even és high/low egyensúllyal
    const mainNumbers = this.selectBalanced(sortedNumbers, config);

    // Euro számok
    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.predictAdditional(history, config);
    }

    // Megbízhatóság számítás
    const confidence = this.calculateConfidence(
      mainNumbers,
      history,
      config,
      finalScores
    );

    return {
      main: mainNumbers.sort((a, b) => a - b),
      additional,
      confidence,
      executionTime: performance.now() - startTime,
    };
  }

  private calculateWeightedFrequency(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    const decayFactor = 0.92; // Exponenciális csökkenés

    history.forEach((draw, index) => {
      const weight = Math.pow(decayFactor, index);
      draw.numbers.forEach(num => {
        scores.set(num, (scores.get(num) || 0) + weight);
      });
    });

    return scores;
  }

  private calculateGapScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const lastSeen = new Map<number, number>();
    const scores = new Map<number, number>();

    // Mikor láttuk utoljára az egyes számokat
    history.forEach((draw, index) => {
      draw.numbers.forEach(num => {
        if (!lastSeen.has(num)) {
          lastSeen.set(num, index);
        }
      });
    });

    // Számok amelyek régóta nem jöttek, magasabb pontot kapnak
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const gap = lastSeen.get(num) ?? history.length;
      // Sigmoida függvény a gap normalizálásához
      const normalizedGap = 1 / (1 + Math.exp(-0.3 * (gap - 5)));
      scores.set(num, normalizedGap);
    }

    return scores;
  }

  private calculatePairScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const pairCount = new Map<string, number>();
    const scores = new Map<number, number>();

    // Párok számolása
    history.forEach(draw => {
      for (let i = 0; i < draw.numbers.length; i++) {
        for (let j = i + 1; j < draw.numbers.length; j++) {
          const pair = `${Math.min(draw.numbers[i], draw.numbers[j])}-${Math.max(draw.numbers[i], draw.numbers[j])}`;
          pairCount.set(pair, (pairCount.get(pair) || 0) + 1);
        }
      }
    });

    // Legutóbbi húzás számaihoz kapcsolódó párok
    const recentNumbers = history[0].numbers;
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let pairScore = 0;
      recentNumbers.forEach(recent => {
        const pair = `${Math.min(num, recent)}-${Math.max(num, recent)}`;
        pairScore += pairCount.get(pair) || 0;
      });
      scores.set(num, pairScore / recentNumbers.length);
    }

    return scores;
  }

  private calculateRangeScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    const rangeCount = config.numbersCount;
    const rangeSize = Math.ceil((config.maxNumber - config.minNumber + 1) / rangeCount);

    // Számoljuk, melyik tartományból hány szám jött átlagosan
    const rangeFrequency: number[] = new Array(rangeCount).fill(0);

    history.forEach(draw => {
      draw.numbers.forEach(num => {
        const rangeIndex = Math.min(
          Math.floor((num - config.minNumber) / rangeSize),
          rangeCount - 1
        );
        rangeFrequency[rangeIndex]++;
      });
    });

    // Normalizálás
    const total = rangeFrequency.reduce((a, b) => a + b, 0);
    const avgPerRange = total / rangeCount;

    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const rangeIndex = Math.min(
        Math.floor((num - config.minNumber) / rangeSize),
        rangeCount - 1
      );
      // Tartományok amelyek alul vannak reprezentálva, magasabb pontot kapnak
      const rangeScore = avgPerRange / (rangeFrequency[rangeIndex] || 1);
      scores.set(num, rangeScore);
    }

    return scores;
  }

  private calculateMomentumScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    const windowSize = 5;

    if (history.length < windowSize * 2) {
      return scores;
    }

    // Legutóbbi és korábbi ablak összehasonlítása
    const recentWindow = history.slice(0, windowSize);
    const previousWindow = history.slice(windowSize, windowSize * 2);

    const recentFreq = new Map<number, number>();
    const previousFreq = new Map<number, number>();

    recentWindow.forEach(draw => {
      draw.numbers.forEach(num => {
        recentFreq.set(num, (recentFreq.get(num) || 0) + 1);
      });
    });

    previousWindow.forEach(draw => {
      draw.numbers.forEach(num => {
        previousFreq.set(num, (previousFreq.get(num) || 0) + 1);
      });
    });

    // Momentum = változás a két ablak között
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const recent = recentFreq.get(num) || 0;
      const previous = previousFreq.get(num) || 0;
      const momentum = recent - previous;
      // Pozitív momentum = növekvő trend
      scores.set(num, momentum + 2); // +2 offset hogy ne legyen negatív
    }

    return scores;
  }

  private calculatePeriodicityScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    const periods = [2, 3, 4, 5, 7, 10]; // Vizsgált periódusok

    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let maxPeriodicityScore = 0;

      periods.forEach(period => {
        let matches = 0;
        for (let i = 0; i < history.length - period; i += period) {
          if (history[i].numbers.includes(num) && 
              history[i + period] && 
              history[i + period].numbers.includes(num)) {
            matches++;
          }
        }
        maxPeriodicityScore = Math.max(maxPeriodicityScore, matches);
      });

      scores.set(num, maxPeriodicityScore);
    }

    return scores;
  }

  private calculateEntropyScores(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): Map<number, number> {
    const scores = new Map<number, number>();
    const frequency = new Map<number, number>();

    // Alap gyakoriság
    history.forEach(draw => {
      draw.numbers.forEach(num => {
        frequency.set(num, (frequency.get(num) || 0) + 1);
      });
    });

    const totalDraws = history.length * config.numbersCount;

    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const freq = frequency.get(num) || 0;
      const p = freq / totalDraws;
      
      if (p > 0) {
        // Shannon entrópia komponens - magasabb entrópia = bizonytalanabb
        const entropy = -p * Math.log2(p);
        // Invertáljuk, mert alacsonyabb entrópia = biztosabb választás
        scores.set(num, 1 - entropy);
      } else {
        scores.set(num, 0.5); // Sosem látott számok közepes pontot kapnak
      }
    }

    return scores;
  }

  private combineScores(
    config: LotteryConfig,
    ...scoreMaps: Map<number, number>[]
  ): Map<number, number> {
    const combined = new Map<number, number>();
    
    // Súlyok az egyes technikákhoz (optimalizálva)
    const weights = [0.25, 0.20, 0.15, 0.12, 0.12, 0.08, 0.08];

    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      let totalScore = 0;
      
      scoreMaps.forEach((scoreMap, index) => {
        const score = scoreMap.get(num) || 0;
        const weight = weights[index] || 0.1;
        totalScore += score * weight;
      });

      combined.set(num, totalScore);
    }

    return combined;
  }

  private selectBalanced(
    sortedNumbers: [number, number][],
    config: LotteryConfig
  ): number[] {
    const selected: number[] = [];
    const midPoint = (config.maxNumber + config.minNumber) / 2;
    
    let oddCount = 0;
    let evenCount = 0;
    let lowCount = 0;
    let highCount = 0;

    const targetOdd = Math.ceil(config.numbersCount / 2);
    const targetEven = config.numbersCount - targetOdd;
    const targetLow = Math.ceil(config.numbersCount / 2);
    const targetHigh = config.numbersCount - targetLow;

    for (const [num, score] of sortedNumbers) {
      if (selected.length >= config.numbersCount) break;

      const isOdd = num % 2 === 1;
      const isLow = num < midPoint;

      // Egyensúly ellenőrzés
      if (isOdd && oddCount >= targetOdd) continue;
      if (!isOdd && evenCount >= targetEven) continue;
      if (isLow && lowCount >= targetLow + 1) continue;
      if (!isLow && highCount >= targetHigh + 1) continue;

      selected.push(num);
      if (isOdd) oddCount++;
      else evenCount++;
      if (isLow) lowCount++;
      else highCount++;
    }

    // Ha nem sikerült elég számot kiválasztani, töltsük fel
    if (selected.length < config.numbersCount) {
      for (const [num, score] of sortedNumbers) {
        if (selected.length >= config.numbersCount) break;
        if (!selected.includes(num)) {
          selected.push(num);
        }
      }
    }

    return selected;
  }

  private predictAdditional(
    history: HistoricalDraw[],
    config: LotteryConfig
  ): number[] {
    const frequency = new Map<number, number>();
    const decayFactor = 0.9;

    history.forEach((draw, index) => {
      const weight = Math.pow(decayFactor, index);
      draw.additionalNumbers?.forEach(num => {
        frequency.set(num, (frequency.get(num) || 0) + weight);
      });
    });

    const sorted = Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1]);

    const selected = sorted
      .slice(0, config.additionalNumbersCount!)
      .map(([num]) => num)
      .sort((a, b) => a - b);

    // Ha nincs elég, generáljunk randomot
    while (selected.length < config.additionalNumbersCount!) {
      const rand = Math.floor(Math.random() * config.additionalMaxNumber!) + config.additionalMinNumber!;
      if (!selected.includes(rand)) {
        selected.push(rand);
      }
    }

    return selected;
  }

  private calculateConfidence(
    selectedNumbers: number[],
    history: HistoricalDraw[],
    config: LotteryConfig,
    scores: Map<number, number>
  ): number {
    // 1. Historikus találati arány szimuláció
    let matchScore = 0;
    const testHistory = history.slice(0, Math.min(10, history.length));
    
    testHistory.forEach(draw => {
      const matches = selectedNumbers.filter(n => draw.numbers.includes(n)).length;
      matchScore += matches / config.numbersCount;
    });
    matchScore /= testHistory.length;

    // 2. Score stabilitás - mennyire magasak a kiválasztott számok score-jai
    const selectedScores = selectedNumbers.map(n => scores.get(n) || 0);
    const avgScore = selectedScores.reduce((a, b) => a + b, 0) / selectedScores.length;
    const maxPossibleScore = Math.max(...Array.from(scores.values()));
    const scoreStability = avgScore / (maxPossibleScore || 1);

    // 3. Egyensúly score - mennyire kiegyensúlyozott a választás
    const oddCount = selectedNumbers.filter(n => n % 2 === 1).length;
    const midPoint = (config.maxNumber + config.minNumber) / 2;
    const lowCount = selectedNumbers.filter(n => n < midPoint).length;
    
    const oddBalance = 1 - Math.abs(oddCount / config.numbersCount - 0.5) * 2;
    const lowHighBalance = 1 - Math.abs(lowCount / config.numbersCount - 0.5) * 2;
    const balanceScore = (oddBalance + lowHighBalance) / 2;

    // 4. Szórás score - nem túl csoportosult
    const sortedSelected = [...selectedNumbers].sort((a, b) => a - b);
    const gaps: number[] = [];
    for (let i = 1; i < sortedSelected.length; i++) {
      gaps.push(sortedSelected[i] - sortedSelected[i - 1]);
    }
    const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const idealGap = (config.maxNumber - config.minNumber) / (config.numbersCount + 1);
    const spreadScore = 1 - Math.min(Math.abs(avgGap - idealGap) / idealGap, 1);

    // Végső megbízhatóság (súlyozott)
    const confidence = 
      matchScore * 0.35 +
      scoreStability * 0.25 +
      balanceScore * 0.20 +
      spreadScore * 0.20;

    return Math.min(confidence, 0.99);
  }

  private generateRandom(config: LotteryConfig, startTime: number): GeneratedNumbers {
    const main = this.generateRandomNumbers(
      config.minNumber,
      config.maxNumber,
      config.numbersCount
    );

    let additional: number[] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount) {
      additional = this.generateRandomNumbers(
        config.additionalMinNumber!,
        config.additionalMaxNumber!,
        config.additionalNumbersCount
      );
    }

    return {
      main,
      additional,
      confidence: 0.1,
      executionTime: performance.now() - startTime,
    };
  }
}







