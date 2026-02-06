/**
 * Predictor Service
 * Lottó szám predikciók generálása optimalizált algoritmusokkal
 */

import { prisma } from '../config/database';
import { logger } from '../utils/logger';
import { AppError } from '../middleware/errorHandler';
import { LOTTERY_CONFIGS } from './scraper.service';

interface Draw {
  year: number;
  week: number;
  numbers: number[];
  additionalNumbers?: number[] | null;
}

interface PredictionResult {
  numbers: number[];
  algorithm: string;
  confidence: number;
}

export class PredictorService {
  /**
   * Predikció generálása egy adott lottó típusra
   */
  static async generatePrediction(lotteryType: string, ticketCount: number = 4) {
    const config = LOTTERY_CONFIGS[lotteryType];
    if (!config) {
      throw new AppError(`Unknown lottery type: ${lotteryType}`, 400, 'UNKNOWN_LOTTERY_TYPE');
    }

    // Lottó típus lekérése
    const lottery = await prisma.lotteryType.findUnique({
      where: { name: config.name },
    });

    if (!lottery) {
      throw new AppError(`Lottery ${lotteryType} not found in database. Please download data first.`, 404, 'LOTTERY_NOT_FOUND');
    }

    // Történelmi adatok lekérése
    const winningNumbers = await prisma.winningNumber.findMany({
      where: { lotteryTypeId: lottery.id },
      orderBy: [{ drawYear: 'asc' }, { drawWeek: 'asc' }],
    });

    // Ha van userPicksCount (pl. Kenó: 10), azt használjuk, különben numbersCount
    const numbersToGenerate = config.userPicksCount || config.numbersCount;

    // Ha nincs elég historikus adat, generálunk random számokat a szabályok alapján
    if (winningNumbers.length < 50) {
      logger.info(`Generating random numbers for ${lotteryType} (only ${winningNumbers.length} draws available)`);
      return this.generateRandomPrediction(config, ticketCount, numbersToGenerate, winningNumbers.length);
    }

    const draws: Draw[] = winningNumbers.map(w => ({
      year: w.drawYear,
      week: w.drawWeek,
      numbers: w.numbers,
      additionalNumbers: w.additionalNumbers,
    }));
    
    const predictor = new LotteryPredictor({
      minNumber: config.minNumber,
      maxNumber: config.maxNumber,
      numbersCount: numbersToGenerate,
    }, draws);

    const allPredictions = predictor.runAllAlgorithms();
    const tickets = predictor.selectBestTickets(allPredictions, ticketCount);

    // Kiegészítő számok generálása (pl. Eurojackpot: 2 szám 1-12 között)
    let additionalNumbersForTickets: number[][] | undefined;
    if (config.hasAdditionalNumbers && config.additionalNumbersCount && config.additionalMaxNumber) {
      // Kiegészítő számok előzményeinek kinyerése
      const additionalDraws = draws
        .filter(d => d.additionalNumbers && d.additionalNumbers.length > 0)
        .map(d => d.additionalNumbers as number[]);
      
      if (additionalDraws.length > 0) {
        const additionalPredictor = new LotteryPredictor({
          minNumber: config.additionalMinNumber || 1,
          maxNumber: config.additionalMaxNumber,
          numbersCount: config.additionalNumbersCount,
        }, additionalDraws.map((nums, i) => ({
          year: draws[i]?.year || 2020,
          week: draws[i]?.week || 1,
          numbers: nums,
        })));
        
        const additionalPredictions = additionalPredictor.runAllAlgorithms();
        const additionalTickets = additionalPredictor.selectBestTickets(additionalPredictions, ticketCount);
        additionalNumbersForTickets = additionalTickets.map(t => t.numbers);
      } else {
        // Ha nincs előzmény, generálunk véletlenszerűen
        additionalNumbersForTickets = tickets.map(() => {
          const nums: number[] = [];
          while (nums.length < (config.additionalNumbersCount || 2)) {
            const n = Math.floor(Math.random() * (config.additionalMaxNumber! - (config.additionalMinNumber || 1) + 1)) + (config.additionalMinNumber || 1);
            if (!nums.includes(n)) nums.push(n);
          }
          return nums.sort((a, b) => a - b);
        });
      }
    }

    // Top számok
    const topNumbers = predictor.getTopNumbers(allPredictions, 15);

    const lastDraw = draws[draws.length - 1];

    return {
      lotteryType: config.displayName,
      totalDrawsAnalyzed: draws.length,
      numbersCount: numbersToGenerate, // Hány számot generálunk/tippel a felhasználó
      maxNumber: config.maxNumber,
      drawnCount: config.numbersCount, // Hány számot húznak a sorsoláson
      hasAdditionalNumbers: config.hasAdditionalNumbers,
      additionalNumbersCount: config.additionalNumbersCount,
      additionalMaxNumber: config.additionalMaxNumber,
      // Új mezők
      country: config.country,
      countryCode: config.countryCode,
      emoji: config.emoji,
      playDomain: config.playDomain,
      drawTime: config.drawTime,
      drawDays: config.drawDays,
      lastDraw: {
        year: lastDraw.year,
        week: lastDraw.week,
        numbers: lastDraw.numbers,
        additionalNumbers: lastDraw.additionalNumbers,
      },
      tickets: tickets.map((t, i) => ({
        ticketNumber: i + 1,
        numbers: t.numbers,
        additionalNumbers: additionalNumbersForTickets ? additionalNumbersForTickets[i] : undefined,
        algorithm: t.algorithm,
        confidence: Math.round(t.confidence * 100) / 100,
      })),
      topNumbers: topNumbers.map(([num, score]) => ({
        number: num,
        score: Math.round(score * 100) / 100,
      })),
      generatedAt: new Date().toISOString(),
    };
  }

  /**
   * Random számok generálása, ha nincs elég historikus adat
   * (Nemzetközi lottókhoz, ahol még nincs scraper)
   */
  private static generateRandomPrediction(
    config: typeof LOTTERY_CONFIGS[keyof typeof LOTTERY_CONFIGS], 
    ticketCount: number, 
    numbersToGenerate: number,
    existingDrawsCount: number
  ) {
    const tickets: { numbers: number[]; additionalNumbers?: number[]; algorithm: string; confidence: number }[] = [];
    const algorithms = ['RandomBalanced', 'RandomSpread', 'RandomHotCold', 'RandomPattern'];
    
    for (let i = 0; i < ticketCount; i++) {
      const numbers = this.generateRandomNumbers(
        config.minNumber, 
        config.maxNumber, 
        numbersToGenerate
      );
      
      let additionalNumbers: number[] | undefined;
      if (config.hasAdditionalNumbers && config.additionalNumbersCount && config.additionalMaxNumber) {
        additionalNumbers = this.generateRandomNumbers(
          config.additionalMinNumber || 1,
          config.additionalMaxNumber,
          config.additionalNumbersCount
        );
      }
      
      tickets.push({
        numbers,
        additionalNumbers,
        algorithm: algorithms[i % algorithms.length],
        confidence: 0.5 + Math.random() * 0.3, // 50-80% random confidence
      });
    }

    // Random top numbers
    const topNumbers: { number: number; score: number }[] = [];
    const usedNumbers = new Set<number>();
    for (let i = 0; i < 15; i++) {
      let num: number;
      do {
        num = Math.floor(Math.random() * (config.maxNumber - config.minNumber + 1)) + config.minNumber;
      } while (usedNumbers.has(num));
      usedNumbers.add(num);
      topNumbers.push({ number: num, score: Math.round((10 - i * 0.5) * 100) / 100 });
    }

    return {
      lotteryType: config.displayName,
      totalDrawsAnalyzed: existingDrawsCount,
      numbersCount: numbersToGenerate,
      maxNumber: config.maxNumber,
      drawnCount: config.numbersCount,
      hasAdditionalNumbers: config.hasAdditionalNumbers,
      additionalNumbersCount: config.additionalNumbersCount,
      additionalMaxNumber: config.additionalMaxNumber,
      country: config.country,
      countryCode: config.countryCode,
      emoji: config.emoji,
      playDomain: config.playDomain,
      drawTime: config.drawTime,
      drawDays: config.drawDays,
      isRandomGenerated: true, // Jelezzük, hogy random generált
      lastDraw: null,
      tickets: tickets.map((t, i) => ({
        ticketNumber: i + 1,
        numbers: t.numbers,
        additionalNumbers: t.additionalNumbers,
        algorithm: t.algorithm,
        confidence: Math.round(t.confidence * 100) / 100,
      })),
      topNumbers,
      generatedAt: new Date().toISOString(),
    };
  }

  /**
   * Random számok generálása
   */
  private static generateRandomNumbers(min: number, max: number, count: number): number[] {
    const numbers: number[] = [];
    while (numbers.length < count) {
      const n = Math.floor(Math.random() * (max - min + 1)) + min;
      if (!numbers.includes(n)) {
        numbers.push(n);
      }
    }
    return numbers.sort((a, b) => a - b);
  }

  /**
   * Statisztikák lekérése
   */
  static async getStats(lotteryType: string) {
    const config = LOTTERY_CONFIGS[lotteryType];
    if (!config) {
      throw new AppError(`Unknown lottery type: ${lotteryType}`, 400, 'UNKNOWN_LOTTERY_TYPE');
    }

    const lottery = await prisma.lotteryType.findUnique({
      where: { name: config.name },
    });

    if (!lottery) {
      throw new AppError(`Lottery ${lotteryType} not found`, 404, 'LOTTERY_NOT_FOUND');
    }

    const winningNumbers = await prisma.winningNumber.findMany({
      where: { lotteryTypeId: lottery.id },
      orderBy: [{ drawYear: 'desc' }, { drawWeek: 'desc' }],
    });

    // Frekvencia számítás
    const frequency: Map<number, number> = new Map();
    const recentFrequency: Map<number, number> = new Map();
    const lastSeen: Map<number, number> = new Map();

    for (let i = 0; i < winningNumbers.length; i++) {
      for (const num of winningNumbers[i].numbers) {
        frequency.set(num, (frequency.get(num) || 0) + 1);
        if (i < 50) {
          recentFrequency.set(num, (recentFrequency.get(num) || 0) + 1);
        }
        if (!lastSeen.has(num)) {
          lastSeen.set(num, i);
        }
      }
    }

    // Rendezés
    const sortedByFrequency = Array.from(frequency.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    const hotNumbers = sortedByFrequency.slice(0, 10);
    const coldNumbers = Array.from(frequency.entries())
      .sort((a, b) => a[1] - b[1])
      .slice(0, 10);

    // Overdue (esedékes) számok
    const overdueNumbers = Array.from(lastSeen.entries())
      .filter(([_, gap]) => gap > 15)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    const lastDraw = winningNumbers[0];

    return {
      lotteryType: config.displayName,
      totalDraws: winningNumbers.length,
      lastDraw: lastDraw ? {
        year: lastDraw.drawYear,
        week: lastDraw.drawWeek,
        numbers: lastDraw.numbers,
        additionalNumbers: lastDraw.additionalNumbers,
      } : null,
      hotNumbers: hotNumbers.map(([num, freq]) => ({ number: num, frequency: freq })),
      coldNumbers: coldNumbers.map(([num, freq]) => ({ number: num, frequency: freq })),
      overdueNumbers: overdueNumbers.map(([num, gap]) => ({ number: num, drawsSinceSeen: gap })),
      frequencyDistribution: sortedByFrequency.map(([num, freq]) => ({ number: num, frequency: freq })),
    };
  }
}

/**
 * Lottó prediktor osztály az algoritmusokkal
 */
class LotteryPredictor {
  private config: { minNumber: number; maxNumber: number; numbersCount: number };
  private draws: Draw[];
  private freq: Map<number, number> = new Map();
  private pairs: Map<string, number> = new Map();
  private lastSeen: Map<number, number> = new Map();
  private transitions: Map<number, Map<number, number>> = new Map();

  constructor(config: { minNumber: number; maxNumber: number; numbersCount: number }, draws: Draw[]) {
    this.config = config;
    this.draws = draws;
    this.precompute();
  }

  private precompute() {
    for (let i = 0; i < this.draws.length; i++) {
      for (const n of this.draws[i].numbers) {
        this.freq.set(n, (this.freq.get(n) || 0) + 1);
        this.lastSeen.set(n, i);
      }
    }

    for (const d of this.draws) {
      const nums = d.numbers;
      for (let i = 0; i < nums.length; i++) {
        for (let j = i + 1; j < nums.length; j++) {
          const key = `${Math.min(nums[i], nums[j])}-${Math.max(nums[i], nums[j])}`;
          this.pairs.set(key, (this.pairs.get(key) || 0) + 1);
        }
      }
    }

    for (let i = 1; i < this.draws.length; i++) {
      for (const p of this.draws[i - 1].numbers) {
        if (!this.transitions.has(p)) this.transitions.set(p, new Map());
        for (const c of this.draws[i].numbers) {
          this.transitions.get(p)!.set(c, (this.transitions.get(p)!.get(c) || 0) + 1);
        }
      }
    }
  }

  private valid(nums: number[]): number[] {
    const v = [...new Set(nums.filter(n => n >= this.config.minNumber && n <= this.config.maxNumber))];
    if (v.length < this.config.numbersCount) {
      for (const [n] of Array.from(this.freq.entries()).sort((a, b) => b[1] - a[1])) {
        if (!v.includes(n)) {
          v.push(n);
          if (v.length >= this.config.numbersCount) break;
        }
      }
    }
    return v.slice(0, this.config.numbersCount).sort((a, b) => a - b);
  }

  // Gap-alapú algoritmus
  private gapBased(gapMin: number = 15, gapMax: number = 60, freqWeight: number = 0.4): PredictionResult {
    const cur = this.draws.length;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    const scores = new Map<number, number>();

    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const gap = cur - (this.lastSeen.get(n) ?? 0);
      const freq = (this.freq.get(n) || 0) / maxFreq;

      let gapScore = 0;
      if (gap >= gapMin && gap <= gapMax) {
        gapScore = 1 - Math.abs(gap - (gapMin + gapMax) / 2) / ((gapMax - gapMin) / 2);
      } else if (gap > gapMax) {
        gapScore = 0.7;
      }

      scores.set(n, gapScore * (1 - freqWeight) + freq * freqWeight);
    }

    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'Gap-Based', confidence: 0.65 };
  }

  // Páros + Gap kombináció
  private pairGap(): PredictionResult {
    const cur = this.draws.length;
    const pairScores = new Map<number, number>();

    for (const [pair, cnt] of Array.from(this.pairs.entries()).sort((a, b) => b[1] - a[1]).slice(0, 60)) {
      const [a, b] = pair.split('-').map(Number);
      pairScores.set(a, (pairScores.get(a) || 0) + cnt);
      pairScores.set(b, (pairScores.get(b) || 0) + cnt);
    }

    const maxPair = Math.max(...pairScores.values()) || 1;
    const scores = new Map<number, number>();

    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const gap = cur - (this.lastSeen.get(n) ?? 0);
      const pScore = (pairScores.get(n) || 0) / maxPair;
      const gapBonus = gap > 20 && gap < 50 ? 0.3 : 0;
      scores.set(n, pScore + gapBonus);
    }

    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'Pair+Gap', confidence: 0.60 };
  }

  // Markov + Frekvencia
  private markovFreq(): PredictionResult {
    const last = this.draws[this.draws.length - 1].numbers;
    const markovScores = new Map<number, number>();

    for (const n of last) {
      const t = this.transitions.get(n);
      if (t) {
        for (const [next, cnt] of t) {
          if (!last.includes(next)) {
            markovScores.set(next, (markovScores.get(next) || 0) + cnt);
          }
        }
      }
    }

    const maxMarkov = Math.max(...markovScores.values()) || 1;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    const scores = new Map<number, number>();

    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const mScore = (markovScores.get(n) || 0) / maxMarkov;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      scores.set(n, mScore * 0.6 + fScore * 0.4);
    }

    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'Markov+Freq', confidence: 0.58 };
  }

  // Frekvencia alapú
  private frequencyBased(): PredictionResult {
    const numbers = this.valid(
      Array.from(this.freq.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'Frequency', confidence: 0.55 };
  }

  // Recent + Hot
  private recentHot(): PredictionResult {
    const recentFreq = new Map<number, number>();
    for (const d of this.draws.slice(-40)) {
      for (const n of d.numbers) {
        recentFreq.set(n, (recentFreq.get(n) || 0) + 1);
      }
    }

    const maxRecent = Math.max(...recentFreq.values()) || 1;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    const scores = new Map<number, number>();

    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const rScore = (recentFreq.get(n) || 0) / maxRecent;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      scores.set(n, rScore * 0.6 + fScore * 0.4);
    }

    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'Recent+Hot', confidence: 0.57 };
  }

  // Multi-objective
  private multiObjective(): PredictionResult {
    const cur = this.draws.length;
    const maxFreq = Math.max(...this.freq.values()) || 1;

    const recentFreq = new Map<number, number>();
    for (const d of this.draws.slice(-40)) {
      for (const n of d.numbers) {
        recentFreq.set(n, (recentFreq.get(n) || 0) + 1);
      }
    }
    const maxRecent = Math.max(...recentFreq.values()) || 1;

    const pairScores = new Map<number, number>();
    for (const [pair, cnt] of Array.from(this.pairs.entries()).sort((a, b) => b[1] - a[1]).slice(0, 50)) {
      const [a, b] = pair.split('-').map(Number);
      pairScores.set(a, (pairScores.get(a) || 0) + cnt);
      pairScores.set(b, (pairScores.get(b) || 0) + cnt);
    }
    const maxPair = Math.max(...pairScores.values()) || 1;

    const scores = new Map<number, number>();
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      const rScore = (recentFreq.get(n) || 0) / maxRecent;
      const gap = cur - (this.lastSeen.get(n) ?? 0);
      const gScore = gap > 15 && gap < 60 ? 1 : gap > 60 ? 0.7 : 0.3;
      const pScore = (pairScores.get(n) || 0) / maxPair;

      scores.set(n, fScore * 0.3 + rScore * 0.25 + gScore * 0.25 + pScore * 0.2);
    }

    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'MultiObjective', confidence: 0.62 };
  }

  // ==================== ÚJ ALGORITMUSOK ====================

  /**
   * 1. NEURAL PATTERN - Neurális hálózat-szerű mintafelismerés
   * Súlyozott szekvencia-minták keresése az utolsó húzásokból
   */
  private neuralPattern(): PredictionResult {
    const windowSizes = [3, 5, 8, 13]; // Fibonacci-szerű ablakméretek
    const patternScores = new Map<number, number>();
    
    // Minden ablakmérethez elemzés
    for (const ws of windowSizes) {
      if (this.draws.length < ws + 1) continue;
      
      // Jelenlegi ablak számai
      const currentWindow = this.draws.slice(-ws).flatMap(d => d.numbers);
      const windowFreq = new Map<number, number>();
      for (const n of currentWindow) {
        windowFreq.set(n, (windowFreq.get(n) || 0) + 1);
      }
      
      // Hasonló ablak keresése a múltban
      for (let i = ws; i < this.draws.length - 1; i++) {
        const pastWindow = this.draws.slice(i - ws, i).flatMap(d => d.numbers);
        const pastFreq = new Map<number, number>();
        for (const n of pastWindow) {
          pastFreq.set(n, (pastFreq.get(n) || 0) + 1);
        }
        
        // Hasonlóság számítás (Jaccard-szerű)
        let similarity = 0;
        for (const [n, cnt] of windowFreq) {
          if (pastFreq.has(n)) {
            similarity += Math.min(cnt, pastFreq.get(n)!) / Math.max(cnt, pastFreq.get(n)!);
          }
        }
        similarity /= windowFreq.size || 1;
        
        // Ha hasonló, a következő húzás számait súlyozzuk
        if (similarity > 0.3) {
          const nextDraw = this.draws[i];
          for (const n of nextDraw.numbers) {
            patternScores.set(n, (patternScores.get(n) || 0) + similarity * (1 / ws));
          }
        }
      }
    }
    
    const maxScore = Math.max(...patternScores.values()) || 1;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    
    const scores = new Map<number, number>();
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const pScore = (patternScores.get(n) || 0) / maxScore;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      scores.set(n, pScore * 0.7 + fScore * 0.3);
    }
    
    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );
    
    return { numbers, algorithm: 'NeuralPattern', confidence: 0.72 };
  }

  /**
   * 2. CYCLIC PREDICTOR - Ciklikus minták felismerése
   * Bizonyos számok bizonyos időszakokban (év, hónap, hét) gyakrabban jönnek
   */
  private cyclicPredictor(): PredictionResult {
    const currentWeek = this.draws[this.draws.length - 1].week;
    const cycleLengths = [4, 8, 13, 26, 52]; // Heti ciklusok
    const cycleScores = new Map<number, number>();
    
    for (const cycle of cycleLengths) {
      // Számoljuk, mely számok jöttek az aktuális ciklus-pozícióban
      const cyclePosition = currentWeek % cycle;
      
      for (let i = 0; i < this.draws.length; i++) {
        const drawCyclePos = this.draws[i].week % cycle;
        if (Math.abs(drawCyclePos - cyclePosition) <= 1 || 
            Math.abs(drawCyclePos - cyclePosition) >= cycle - 1) {
          // Közel vagyunk a cikluspozícióhoz
          for (const n of this.draws[i].numbers) {
            cycleScores.set(n, (cycleScores.get(n) || 0) + 1 / cycle);
          }
        }
      }
    }
    
    const maxCycle = Math.max(...cycleScores.values()) || 1;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    
    const scores = new Map<number, number>();
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const cScore = (cycleScores.get(n) || 0) / maxCycle;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      scores.set(n, cScore * 0.6 + fScore * 0.4);
    }
    
    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );
    
    return { numbers, algorithm: 'CyclicPredictor', confidence: 0.64 };
  }

  /**
   * 3. DELTA SEQUENCE - Számok közötti különbségek elemzése
   * A kihúzott számok közti távolságok mintázatainak felismerése
   */
  private deltaSequence(): PredictionResult {
    // Delta minták gyűjtése
    const deltaPatterns = new Map<string, number[][]>(); // delta pattern -> [következő számok]
    
    for (let i = 0; i < this.draws.length - 1; i++) {
      const nums = [...this.draws[i].numbers].sort((a, b) => a - b);
      const deltas: number[] = [];
      for (let j = 1; j < nums.length; j++) {
        deltas.push(nums[j] - nums[j - 1]);
      }
      
      // Kategorizált delta pattern (kis/közepes/nagy különbségek)
      const pattern = deltas.map(d => d <= 5 ? 'S' : d <= 15 ? 'M' : 'L').join('');
      
      if (!deltaPatterns.has(pattern)) {
        deltaPatterns.set(pattern, []);
      }
      deltaPatterns.get(pattern)!.push(this.draws[i + 1].numbers);
    }
    
    // Jelenlegi húzás delta mintája
    const lastNums = [...this.draws[this.draws.length - 1].numbers].sort((a, b) => a - b);
    const lastDeltas = [];
    for (let j = 1; j < lastNums.length; j++) {
      lastDeltas.push(lastNums[j] - lastNums[j - 1]);
    }
    const lastPattern = lastDeltas.map(d => d <= 5 ? 'S' : d <= 15 ? 'M' : 'L').join('');
    
    // Hasonló minták utáni számok súlyozása
    const deltaScores = new Map<number, number>();
    
    // Exact match
    if (deltaPatterns.has(lastPattern)) {
      for (const nums of deltaPatterns.get(lastPattern)!) {
        for (const n of nums) {
          deltaScores.set(n, (deltaScores.get(n) || 0) + 2);
        }
      }
    }
    
    // Hasonló minták (1 karakteres eltérés)
    for (const [pattern, numsList] of deltaPatterns) {
      let diff = 0;
      for (let i = 0; i < Math.min(pattern.length, lastPattern.length); i++) {
        if (pattern[i] !== lastPattern[i]) diff++;
      }
      if (diff === 1) {
        for (const nums of numsList) {
          for (const n of nums) {
            deltaScores.set(n, (deltaScores.get(n) || 0) + 1);
          }
        }
      }
    }
    
    const maxDelta = Math.max(...deltaScores.values()) || 1;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    
    const scores = new Map<number, number>();
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const dScore = (deltaScores.get(n) || 0) / maxDelta;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      scores.set(n, dScore * 0.65 + fScore * 0.35);
    }
    
    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );
    
    return { numbers, algorithm: 'DeltaSequence', confidence: 0.66 };
  }

  /**
   * 4. CLUSTER BALANCE - Klaszter-alapú egyensúlyozás
   * A számtartományt klaszterekre osztja és minden klaszterből választ
   */
  private clusterBalance(): PredictionResult {
    const range = this.config.maxNumber - this.config.minNumber + 1;
    const clusterCount = Math.min(this.config.numbersCount + 1, Math.ceil(range / 10));
    const clusterSize = Math.ceil(range / clusterCount);
    
    // Klaszter statisztikák
    const clusterFreq: Map<number, number>[] = [];
    for (let c = 0; c < clusterCount; c++) {
      clusterFreq.push(new Map());
    }
    
    for (const d of this.draws) {
      for (const n of d.numbers) {
        const cluster = Math.min(Math.floor((n - this.config.minNumber) / clusterSize), clusterCount - 1);
        clusterFreq[cluster].set(n, (clusterFreq[cluster].get(n) || 0) + 1);
      }
    }
    
    // Minden klaszterből a legjobb számok
    const selectedNumbers: number[] = [];
    const perCluster = Math.ceil(this.config.numbersCount / clusterCount);
    
    for (let c = 0; c < clusterCount; c++) {
      const sorted = Array.from(clusterFreq[c].entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, perCluster)
        .map(e => e[0]);
      selectedNumbers.push(...sorted);
    }
    
    // Gap-alapú bonus
    const cur = this.draws.length;
    const scores = new Map<number, number>();
    
    for (const n of selectedNumbers) {
      const freq = this.freq.get(n) || 0;
      const gap = cur - (this.lastSeen.get(n) ?? 0);
      const gapBonus = gap > 10 && gap < 40 ? 0.2 : 0;
      scores.set(n, freq + gapBonus * 100);
    }
    
    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );
    
    return { numbers, algorithm: 'ClusterBalance', confidence: 0.67 };
  }

  /**
   * 5. MOMENTUM TRACKER - Momentum alapú trend követés
   * Követi mely számok gyakorisága nő vagy csökken
   */
  private momentumTracker(): PredictionResult {
    const recentWindow = 20;
    const olderWindow = 60;
    
    // Régebbi időszak frekvenciája
    const olderFreq = new Map<number, number>();
    const olderDraws = this.draws.slice(-(olderWindow + recentWindow), -recentWindow);
    for (const d of olderDraws) {
      for (const n of d.numbers) {
        olderFreq.set(n, (olderFreq.get(n) || 0) + 1);
      }
    }
    
    // Közelmúlt frekvenciája
    const recentFreq = new Map<number, number>();
    const recentDraws = this.draws.slice(-recentWindow);
    for (const d of recentDraws) {
      for (const n of d.numbers) {
        recentFreq.set(n, (recentFreq.get(n) || 0) + 1);
      }
    }
    
    // Momentum számítás (normalizált változás)
    const momentum = new Map<number, number>();
    const maxOlder = Math.max(...olderFreq.values()) || 1;
    const maxRecent = Math.max(...recentFreq.values()) || 1;
    
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const oldNorm = (olderFreq.get(n) || 0) / maxOlder;
      const recentNorm = (recentFreq.get(n) || 0) / maxRecent;
      
      // Pozitív momentum = szám gyakorisága növekszik
      const mom = recentNorm - oldNorm * 0.7; // Kissé a régebbi felé súlyozva
      momentum.set(n, mom);
    }
    
    // Kombináljuk a momentumot az általános frekvenciával és gap-pel
    const cur = this.draws.length;
    const maxFreq = Math.max(...this.freq.values()) || 1;
    
    const scores = new Map<number, number>();
    for (let n = this.config.minNumber; n <= this.config.maxNumber; n++) {
      const mom = momentum.get(n) || 0;
      const fScore = (this.freq.get(n) || 0) / maxFreq;
      const gap = cur - (this.lastSeen.get(n) ?? 0);
      const gapScore = gap > 8 && gap < 30 ? 0.3 : gap > 30 && gap < 60 ? 0.2 : 0;
      
      // Pozitív momentum + gyakori + megfelelő gap = jó jelölt
      scores.set(n, Math.max(0, mom) * 0.4 + fScore * 0.4 + gapScore);
    }
    
    const numbers = this.valid(
      Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );
    
    return { numbers, algorithm: 'MomentumTracker', confidence: 0.69 };
  }

  // Ensemble - Most már az összes algoritmust kombinálja
  private ensemble(): PredictionResult {
    const predictions = [
      this.gapBased(),
      this.pairGap(),
      this.markovFreq(),
      this.frequencyBased(),
      this.recentHot(),
      this.multiObjective(),
      // Új algoritmusok
      this.neuralPattern(),
      this.cyclicPredictor(),
      this.deltaSequence(),
      this.clusterBalance(),
      this.momentumTracker(),
    ];

    const votes = new Map<number, number>();
    for (const pred of predictions) {
      for (const n of pred.numbers) {
        // Súlyozott szavazás a confidence alapján
        votes.set(n, (votes.get(n) || 0) + pred.confidence);
      }
    }

    const numbers = this.valid(
      Array.from(votes.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    // Magasabb confidence, mivel több algoritmus konszenzusa
    return { numbers, algorithm: 'Ensemble', confidence: 0.75 };
  }

  // SUPER ENSEMBLE - Top 5 algoritmus súlyozott kombinációja
  private superEnsemble(): PredictionResult {
    const allPreds = [
      this.neuralPattern(),
      this.momentumTracker(),
      this.clusterBalance(),
      this.deltaSequence(),
      this.multiObjective(),
    ];

    // Súlyozott szavazás
    const votes = new Map<number, number>();
    const weights = [1.2, 1.1, 1.0, 1.0, 0.9]; // NeuralPattern és Momentum kapnak extra súlyt
    
    for (let i = 0; i < allPreds.length; i++) {
      const pred = allPreds[i];
      const weight = weights[i];
      for (let j = 0; j < pred.numbers.length; j++) {
        const n = pred.numbers[j];
        // Pozíció alapú súlyozás is (első számok fontosabbak)
        const posWeight = 1 - (j * 0.05);
        votes.set(n, (votes.get(n) || 0) + pred.confidence * weight * posWeight);
      }
    }

    const numbers = this.valid(
      Array.from(votes.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0])
    );

    return { numbers, algorithm: 'SuperEnsemble', confidence: 0.78 };
  }

  runAllAlgorithms(): PredictionResult[] {
    return [
      // Klasszikus algoritmusok
      this.gapBased(15, 60, 0.4),
      this.pairGap(),
      this.markovFreq(),
      this.frequencyBased(),
      this.recentHot(),
      this.multiObjective(),
      // Új fejlett algoritmusok
      this.neuralPattern(),
      this.cyclicPredictor(),
      this.deltaSequence(),
      this.clusterBalance(),
      this.momentumTracker(),
      // Ensemble módszerek
      this.ensemble(),
      this.superEnsemble(),
    ];
  }

  selectBestTickets(predictions: PredictionResult[], count: number): PredictionResult[] {
    const seen = new Set<string>();
    const tickets: PredictionResult[] = [];

    // Rendezés confidence szerint
    const sorted = [...predictions].sort((a, b) => b.confidence - a.confidence);

    for (const pred of sorted) {
      const key = pred.numbers.join(',');
      if (!seen.has(key)) {
        seen.add(key);
        tickets.push(pred);
        if (tickets.length >= count) break;
      }
    }

    return tickets;
  }

  getTopNumbers(predictions: PredictionResult[], count: number): [number, number][] {
    const votes = new Map<number, number>();
    for (const pred of predictions) {
      for (const n of pred.numbers) {
        votes.set(n, (votes.get(n) || 0) + pred.confidence);
      }
    }

    return Array.from(votes.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, count);
  }
}
