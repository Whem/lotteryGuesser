import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Seeding database...');

  // Create lottery types
  const hungarian590 = await prisma.lotteryType.upsert({
    where: { name: 'hungarian_5_90' },
    update: {},
    create: {
      name: 'hungarian_5_90',
      displayName: 'Ötöslottó (5/90)',
      country: 'Hungary',
      description: 'Hungarian national lottery - pick 5 numbers from 1-90',
      imageUrl: '/images/otoslotto.png',
      minNumber: 1,
      maxNumber: 90,
      numbersCount: 5,
      hasAdditionalNumbers: false,
      drawDays: ['Saturday'],
      drawTime: '18:35',
      timezone: 'Europe/Budapest',
      currency: 'HUF',
    },
  });

  const eurojackpot = await prisma.lotteryType.upsert({
    where: { name: 'eurojackpot' },
    update: {},
    create: {
      name: 'eurojackpot',
      displayName: 'Eurojackpot',
      country: 'Europe',
      description: 'European lottery - pick 5 numbers from 1-50 and 2 from 1-12',
      imageUrl: '/images/eurojackpot.png',
      minNumber: 1,
      maxNumber: 50,
      numbersCount: 5,
      hasAdditionalNumbers: true,
      additionalMinNumber: 1,
      additionalMaxNumber: 12,
      additionalNumbersCount: 2,
      drawDays: ['Tuesday', 'Friday'],
      drawTime: '20:00',
      timezone: 'Europe/Berlin',
      currency: 'EUR',
    },
  });

  console.log('✅ Lottery types created');

  // Create algorithms
  const algorithms = [
    { name: 'frequency_analysis', displayName: 'Frequency Analysis', description: 'Selects numbers based on their historical frequency', category: 'STATISTICAL', complexity: 'SIMPLE', isPremium: false },
    { name: 'hot_cold_balance', displayName: 'Hot & Cold Balance', description: 'Combines frequently and rarely drawn numbers', category: 'STATISTICAL', complexity: 'SIMPLE', isPremium: false },
    { name: 'gap_analysis', displayName: 'Gap Analysis', description: 'Selects numbers that are "due" based on gaps', category: 'STATISTICAL', complexity: 'MODERATE', isPremium: false },
    { name: 'weighted_frequency', displayName: 'Weighted Frequency', description: 'Recent draws weighted more heavily', category: 'STATISTICAL', complexity: 'MODERATE', isPremium: false },
    { name: 'positional_frequency', displayName: 'Positional Frequency', description: 'Position-specific frequency analysis', category: 'STATISTICAL', complexity: 'MODERATE', isPremium: true },
    { name: 'sum_range_optimization', displayName: 'Sum Range Optimizer', description: 'Optimizes for historical sum ranges', category: 'STATISTICAL', complexity: 'MODERATE', isPremium: false },
    { name: 'odd_even_balance', displayName: 'Odd/Even Balance', description: 'Optimal odd/even ratio based on history', category: 'STATISTICAL', complexity: 'SIMPLE', isPremium: false },
    { name: 'consecutive_pairs', displayName: 'Consecutive Pairs', description: 'Number pair co-occurrence analysis', category: 'STATISTICAL', complexity: 'MODERATE', isPremium: false },
    { name: 'markov_chain', displayName: 'Markov Chain', description: 'Transition probabilities between numbers', category: 'PROBABILITY', complexity: 'COMPLEX', isPremium: true },
    { name: 'bayesian_prediction', displayName: 'Bayesian Inference', description: 'Bayesian probability updates', category: 'PROBABILITY', complexity: 'COMPLEX', isPremium: true },
    { name: 'monte_carlo', displayName: 'Monte Carlo Simulation', description: 'Random sampling simulation', category: 'PROBABILITY', complexity: 'COMPLEX', isPremium: false },
    { name: 'pattern_recognition', displayName: 'Pattern Recognition', description: 'Identifies recurring patterns', category: 'PATTERN_RECOGNITION', complexity: 'COMPLEX', isPremium: true },
    { name: 'cyclic_pattern', displayName: 'Cyclic Pattern Detection', description: 'Detects repeating cycles', category: 'PATTERN_RECOGNITION', complexity: 'MODERATE', isPremium: false },
    { name: 'cluster_analysis', displayName: 'Cluster Analysis', description: 'Groups numbers into clusters', category: 'PATTERN_RECOGNITION', complexity: 'MODERATE', isPremium: false },
    { name: 'sequence_detection', displayName: 'Sequence Detection', description: 'Identifies arithmetic sequences', category: 'PATTERN_RECOGNITION', complexity: 'MODERATE', isPremium: false },
    { name: 'neural_network', displayName: 'Neural Network', description: 'Simple neural network prediction', category: 'MACHINE_LEARNING', complexity: 'ADVANCED', isPremium: true },
    { name: 'random_forest', displayName: 'Random Forest', description: 'Ensemble of decision trees', category: 'MACHINE_LEARNING', complexity: 'ADVANCED', isPremium: true },
    { name: 'genetic_algorithm', displayName: 'Genetic Algorithm', description: 'Evolves optimal combinations', category: 'MACHINE_LEARNING', complexity: 'ADVANCED', isPremium: true },
    { name: 'ensemble_voting', displayName: 'Ensemble Voting', description: 'Democratic voting across strategies', category: 'ENSEMBLE', complexity: 'COMPLEX', isPremium: false },
    { name: 'weighted_ensemble', displayName: 'Weighted Ensemble', description: 'Performance-weighted combination', category: 'ENSEMBLE', complexity: 'COMPLEX', isPremium: true },
    { name: 'lucky_random', displayName: 'Lucky Random', description: 'Random with slight frequency bias', category: 'EXPERIMENTAL', complexity: 'SIMPLE', isPremium: false },
    { name: 'fibonacci_sequence', displayName: 'Fibonacci Sequence', description: 'Uses Fibonacci relationships', category: 'EXPERIMENTAL', complexity: 'MODERATE', isPremium: false },
    // NEW - Hybrid Advanced
    { name: 'adaptive_resonance', displayName: 'Adaptive Resonance', description: 'Neural-inspired resonance pattern detection', category: 'HYBRID', complexity: 'ADVANCED', isPremium: false },
    { name: 'quantum_walk', displayName: 'Quantum Walk', description: 'Quantum superposition and interference patterns', category: 'HYBRID', complexity: 'ADVANCED', isPremium: false },
    { name: 'swarm_intelligence', displayName: 'Swarm Intelligence', description: 'Ant colony optimization with multiple agents', category: 'HYBRID', complexity: 'ADVANCED', isPremium: false },
    { name: 'temporal_convolution', displayName: 'Temporal Convolution', description: 'CNN-inspired convolution filters on time series', category: 'MACHINE_LEARNING', complexity: 'ADVANCED', isPremium: false },
    { name: 'gravitational_field', displayName: 'Gravitational Field', description: 'Physics simulation with gravitational attraction', category: 'HYBRID', complexity: 'ADVANCED', isPremium: false },
  ];

  for (const algo of algorithms) {
    await prisma.algorithm.upsert({
      where: { name: algo.name },
      update: algo,
      create: algo as any,
    });
  }

  console.log('✅ Algorithms created');

  // Add sample winning numbers for Hungarian 5/90
  const hungarian590Numbers = [
    { numbers: [7, 23, 45, 67, 89], date: '2024-11-23' },
    { numbers: [3, 18, 34, 56, 78], date: '2024-11-16' },
    { numbers: [12, 29, 41, 63, 85], date: '2024-11-09' },
    { numbers: [5, 21, 38, 52, 71], date: '2024-11-02' },
    { numbers: [9, 27, 44, 66, 88], date: '2024-10-26' },
    { numbers: [14, 33, 47, 59, 82], date: '2024-10-19' },
    { numbers: [2, 19, 36, 54, 73], date: '2024-10-12' },
    { numbers: [8, 25, 42, 61, 79], date: '2024-10-05' },
    { numbers: [11, 28, 39, 57, 86], date: '2024-09-28' },
    { numbers: [6, 22, 35, 50, 68], date: '2024-09-21' },
  ];

  for (const draw of hungarian590Numbers) {
    const drawDate = new Date(draw.date);
    const year = drawDate.getFullYear();
    const startOfYear = new Date(year, 0, 1);
    const days = Math.floor((drawDate.getTime() - startOfYear.getTime()) / (24 * 60 * 60 * 1000));
    const week = Math.ceil((days + startOfYear.getDay() + 1) / 7);

    await prisma.winningNumber.upsert({
      where: {
        lotteryTypeId_drawYear_drawWeek: {
          lotteryTypeId: hungarian590.id,
          drawYear: year,
          drawWeek: week,
        },
      },
      update: { numbers: draw.numbers },
      create: {
        lotteryTypeId: hungarian590.id,
        numbers: draw.numbers,
        drawDate,
        drawYear: year,
        drawWeek: week,
      },
    });
  }

  console.log('✅ Hungarian 5/90 winning numbers added');

  // Add sample winning numbers for Eurojackpot (updated December 2024)
  const eurojackpotNumbers = [
    // December 2024 - legfrissebb húzások
    { numbers: [2, 25, 27, 37, 50], additional: [2, 11], date: '2024-12-13' },
    { numbers: [3, 16, 23, 34, 49], additional: [4, 8], date: '2024-12-10' },
    { numbers: [8, 14, 29, 41, 47], additional: [1, 10], date: '2024-12-06' },
    { numbers: [5, 19, 33, 38, 45], additional: [3, 12], date: '2024-12-03' },
    // November 2024
    { numbers: [11, 22, 30, 44, 48], additional: [5, 9], date: '2024-11-29' },
    { numbers: [7, 17, 26, 35, 50], additional: [2, 7], date: '2024-11-26' },
    { numbers: [4, 15, 28, 37, 49], additional: [3, 9], date: '2024-11-22' },
    { numbers: [7, 19, 31, 42, 48], additional: [5, 11], date: '2024-11-19' },
    { numbers: [2, 12, 25, 38, 46], additional: [1, 8], date: '2024-11-15' },
    { numbers: [9, 21, 33, 44, 50], additional: [4, 10], date: '2024-11-12' },
    { numbers: [5, 17, 29, 40, 47], additional: [2, 7], date: '2024-11-08' },
    { numbers: [1, 14, 26, 35, 45], additional: [6, 12], date: '2024-11-05' },
    { numbers: [8, 20, 32, 41, 49], additional: [3, 9], date: '2024-11-01' },
    // October 2024
    { numbers: [3, 16, 27, 39, 48], additional: [5, 11], date: '2024-10-29' },
    { numbers: [6, 18, 30, 43, 50], additional: [1, 8], date: '2024-10-25' },
    { numbers: [10, 22, 34, 46, 47], additional: [4, 10], date: '2024-10-22' },
    { numbers: [13, 24, 36, 42, 49], additional: [2, 6], date: '2024-10-18' },
    { numbers: [4, 11, 28, 39, 45], additional: [3, 11], date: '2024-10-15' },
    { numbers: [7, 20, 31, 44, 48], additional: [1, 9], date: '2024-10-11' },
    { numbers: [2, 15, 25, 37, 50], additional: [5, 12], date: '2024-10-08' },
  ];

  for (const draw of eurojackpotNumbers) {
    const drawDate = new Date(draw.date);
    const year = drawDate.getFullYear();
    const startOfYear = new Date(year, 0, 1);
    const days = Math.floor((drawDate.getTime() - startOfYear.getTime()) / (24 * 60 * 60 * 1000));
    const week = Math.ceil((days + startOfYear.getDay() + 1) / 7);

    await prisma.winningNumber.upsert({
      where: {
        lotteryTypeId_drawYear_drawWeek: {
          lotteryTypeId: eurojackpot.id,
          drawYear: year,
          drawWeek: week,
        },
      },
      update: { numbers: draw.numbers, additionalNumbers: draw.additional },
      create: {
        lotteryTypeId: eurojackpot.id,
        numbers: draw.numbers,
        additionalNumbers: draw.additional,
        drawDate,
        drawYear: year,
        drawWeek: week,
      },
    });
  }

  console.log('✅ Eurojackpot winning numbers added');
  console.log('🎉 Seeding completed!');
}

main()
  .catch((e) => {
    console.error('❌ Seeding failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
