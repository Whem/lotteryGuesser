// Common types for the lottery system

export interface LotteryConfig {
  minNumber: number;
  maxNumber: number;
  numbersCount: number;
  hasAdditionalNumbers: boolean;
  additionalMinNumber?: number;
  additionalMaxNumber?: number;
  additionalNumbersCount?: number;
  // Aliases for backwards compatibility
  hasadditional?: boolean;
  additionalCount?: number;
}

export interface GeneratedNumbers {
  main: number[];
  additional?: number[];
  confidence?: number;
  executionTime: number;
}

export interface AlgorithmResult {
  algorithmId: string;
  algorithmName: string;
  numbers: GeneratedNumbers;
  metadata?: Record<string, any>;
}

export interface CompetitionResult {
  algorithmId: string;
  algorithmName: string;
  matchedMain: number;
  matchedAdditional: number;
  score: number;
  rank: number;
}

export interface PaginationParams {
  page: number;
  limit: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

// Historical data for algorithms
export interface HistoricalDraw {
  numbers: number[];
  additionalNumbers?: number[];
  additional?: number[]; // Alias for backwards compatibility
  drawDate: Date;
  year: number;
  week: number;
}

// Algorithm interface that all prediction algorithms must implement
export interface PredictionAlgorithm {
  name: string;
  displayName: string;
  description: string;
  category: 'STATISTICAL' | 'MACHINE_LEARNING' | 'PATTERN_RECOGNITION' | 'PROBABILITY' | 'ENSEMBLE' | 'HYBRID' | 'EXPERIMENTAL';
  complexity: 'SIMPLE' | 'MODERATE' | 'COMPLEX' | 'ADVANCED';
  isPremium: boolean;
  
  predict(
    config: LotteryConfig,
    history: HistoricalDraw[],
    options?: Record<string, any>
  ): Promise<GeneratedNumbers>;
}
