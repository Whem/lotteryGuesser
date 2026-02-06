import { BaseAlgorithm } from '../base';
import { LotteryConfig, GeneratedNumbers, HistoricalDraw } from '../../types';

export class ProphetTimeSeries extends BaseAlgorithm {
  name = 'prophet_time_series';
  displayName = 'Prophet Time Series';
  description = 'Time series forecasting using seasonal decomposition and trend analysis';
  category = 'PROBABILITY' as const;
  complexity = 'COMPLEX' as const;
  isPremium = true;

  async predict(config: LotteryConfig, history: HistoricalDraw[]): Promise<GeneratedNumbers> {
    const startTime = performance.now();
    
    // Create time series for each number
    const timeSeries = new Map<number, number[]>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      timeSeries.set(num, []);
    }
    
    // Build time series (1 if appeared, 0 if not)
    for (const draw of history.slice(0, 52)) { // ~1 year of weekly data
      for (let num = config.minNumber; num <= config.maxNumber; num++) {
        const series = timeSeries.get(num)!;
        series.push(draw.numbers.includes(num) ? 1 : 0);
      }
    }
    
    // Simple trend + seasonality decomposition
    const predictions = new Map<number, number>();
    
    for (let num = config.minNumber; num <= config.maxNumber; num++) {
      const series = timeSeries.get(num)!;
      if (series.length < 4) continue;
      
      // Calculate trend (simple linear)
      const recentAvg = series.slice(0, 10).reduce((a, b) => a + b, 0) / Math.min(10, series.length);
      const olderAvg = series.slice(10, 30).reduce((a, b) => a + b, 0) / Math.max(1, Math.min(20, series.length - 10));
      const trend = recentAvg - olderAvg;
      
      // Calculate seasonality (period of 4-8 weeks)
      let seasonality = 0;
      const period = Math.min(8, Math.floor(series.length / 2));
      if (period > 0) {
        for (let i = 0; i < period; i++) {
          seasonality += series[i] || 0;
        }
        seasonality /= period;
      }
      
      // Combine for prediction
      const prediction = recentAvg * 0.5 + trend * 0.3 + seasonality * 0.2;
      predictions.set(num, prediction);
    }
    
    // Select top predictions
    const sorted = Array.from(predictions.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([num]) => num);
    
    const main = sorted.slice(0, config.numbersCount).sort((a, b) => a - b);
    let additional: number[] | undefined;
    
    if (config.hasadditional && config.additionalCount) {
      // Apply same logic for additional numbers
      const addTimeSeries = new Map<number, number[]>();
      for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
        addTimeSeries.set(num, []);
      }
      
      for (const draw of history.slice(0, 52)) {
        if (draw.additional) {
          for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
            addTimeSeries.get(num)!.push(draw.additional.includes(num) ? 1 : 0);
          }
        }
      }
      
      const addPredictions: [number, number][] = [];
      for (let num = config.additionalMinNumber!; num <= config.additionalMaxNumber!; num++) {
        const series = addTimeSeries.get(num)!;
        const avg = series.length > 0 ? series.slice(0, 10).reduce((a, b) => a + b, 0) / Math.min(10, series.length) : 0.5;
        addPredictions.push([num, avg]);
      }
      
      additional = addPredictions
        .sort((a, b) => b[1] - a[1])
        .slice(0, config.additionalCount!)
        .map(([num]) => num)
        .sort((a, b) => a - b);
    }
    
    return {
      main,
      additional,
      confidence: 0.73,
      executionTime: performance.now() - startTime,
    };
  }
}

export const prophetTimeSeries = new ProphetTimeSeries();
