import { PredictionAlgorithm } from '../types';

// Import all algorithms - Statistical
import { FrequencyAnalysis } from './statistical/frequencyAnalysis';
import { HotColdBalance } from './statistical/hotColdBalance';
import { GapAnalysis } from './statistical/gapAnalysis';
import { WeightedFrequency } from './statistical/weightedFrequency';
import { PositionalFrequency } from './statistical/positionalFrequency';
import { SumRangeOptimization } from './statistical/sumRangeOptimization';
import { OddEvenBalance } from './statistical/oddEvenBalance';
import { ConsecutivePairs } from './statistical/consecutivePairs';
import { PrimeCompositeBalance } from './statistical/primeCompositeBalance';
import { TrendWeightedGap } from './statistical/trendWeightedGap';

// Probability
import { MarkovChain } from './probability/markovChain';
import { BayesianPrediction } from './probability/bayesianPrediction';
import { MonteCarlo } from './probability/monteCarlo';
import { RandomWalkMomentum } from './probability/randomWalkMomentum';
import { ProphetTimeSeries } from './probability/prophetTimeSeries';

// Pattern Recognition
import { PatternRecognition } from './pattern/patternRecognition';
import { CyclicPattern } from './pattern/cyclicPattern';
import { ClusterAnalysis } from './pattern/clusterAnalysis';
import { SequenceDetection } from './pattern/sequenceDetection';
import { AlternatingClusterBand } from './pattern/alternatingClusterBand';
import { BalancedEntropyContrast } from './pattern/balancedEntropyContrast';
import { DrawClusterRegularity } from './pattern/drawClusterRegularity';

// Machine Learning
import { NeuralNetwork } from './ml/neuralNetwork';
import { RandomForest } from './ml/randomForest';
import { GeneticAlgorithm } from './ml/geneticAlgorithm';
import { SlidingWindowEntropy } from './ml/slidingWindowEntropy';

// Ensemble
import { EnsembleVoting } from './ensemble/ensembleVoting';
import { WeightedEnsemble } from './ensemble/weightedEnsemble';

// Experimental
import { LuckyRandom } from './experimental/luckyRandom';
import { FibonacciSequence } from './experimental/fibonacciSequence';
import { EqualIntervalSpacing } from './experimental/equalIntervalSpacing';
import { HierarchicalRegime } from './experimental/hierarchicalRegime';

// Hybrid (NEW - advanced algorithms)
import { AdaptiveResonance } from './hybrid/adaptiveResonance';
import { QuantumWalk } from './hybrid/quantumWalk';
import { SwarmIntelligence } from './hybrid/swarmIntelligence';
import { TemporalConvolution } from './hybrid/temporalConvolution';
import { GravitationalField } from './hybrid/gravitationalField';

class AlgorithmRegistry {
  private algorithms: Map<string, PredictionAlgorithm> = new Map();

  register(algorithm: PredictionAlgorithm) {
    this.algorithms.set(algorithm.name, algorithm);
  }

  get(name: string): PredictionAlgorithm | undefined {
    return this.algorithms.get(name);
  }

  getAll(): PredictionAlgorithm[] {
    return Array.from(this.algorithms.values());
  }

  getByCategory(category: string): PredictionAlgorithm[] {
    return this.getAll().filter(a => a.category === category);
  }
}

export const algorithmRegistry = new AlgorithmRegistry();

// Register all algorithms
const allAlgorithms: PredictionAlgorithm[] = [
  // Statistical (10)
  new FrequencyAnalysis(),
  new HotColdBalance(),
  new GapAnalysis(),
  new WeightedFrequency(),
  new PositionalFrequency(),
  new SumRangeOptimization(),
  new OddEvenBalance(),
  new ConsecutivePairs(),
  new PrimeCompositeBalance(),
  new TrendWeightedGap(),
  
  // Probability (5)
  new MarkovChain(),
  new BayesianPrediction(),
  new MonteCarlo(),
  new RandomWalkMomentum(),
  new ProphetTimeSeries(),
  
  // Pattern Recognition (7)
  new PatternRecognition(),
  new CyclicPattern(),
  new ClusterAnalysis(),
  new SequenceDetection(),
  new AlternatingClusterBand(),
  new BalancedEntropyContrast(),
  new DrawClusterRegularity(),
  
  // Machine Learning (4)
  new NeuralNetwork(),
  new RandomForest(),
  new GeneticAlgorithm(),
  new SlidingWindowEntropy(),
  
  // Ensemble (2)
  new EnsembleVoting(),
  new WeightedEnsemble(),
  
  // Experimental (4)
  new LuckyRandom(),
  new FibonacciSequence(),
  new EqualIntervalSpacing(),
  new HierarchicalRegime(),

  // Hybrid - Advanced (5) NEW
  new AdaptiveResonance(),
  new QuantumWalk(),
  new SwarmIntelligence(),
  new TemporalConvolution(),
  new GravitationalField(),
];

allAlgorithms.forEach(algo => algorithmRegistry.register(algo));

export { AlgorithmRegistry };
