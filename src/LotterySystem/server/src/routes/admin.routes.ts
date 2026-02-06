/**
 * Admin Routes
 * Adminisztratív műveletek - lottó számok letöltése, konfigurálás
 */

import { Router, Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandler';
import { ScraperService, LOTTERY_CONFIGS } from '../services/scraper.service';
import { PredictorService } from '../services/predictor.service';

const router = Router();

/**
 * GET /api/admin/lottery-types
 * Elérhető lottó típusok listázása
 */
router.get(
  '/lottery-types',
  asyncHandler(async (req: Request, res: Response) => {
    const types = ScraperService.getAvailableLotteryTypes();
    res.json({ success: true, data: types });
  })
);

/**
 * POST /api/admin/download/:lotteryType
 * Adott lottó típus számainak letöltése az internetről
 */
router.post(
  '/download/:lotteryType',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryType } = req.params;
    
    if (!LOTTERY_CONFIGS[lotteryType]) {
      return res.status(400).json({
        success: false,
        error: `Unknown lottery type: ${lotteryType}`,
        availableTypes: Object.keys(LOTTERY_CONFIGS),
      });
    }
    
    const result = await ScraperService.downloadAndSaveNumbers(lotteryType);
    
    res.json({
      success: true,
      message: `Downloaded ${result.newDraws} new draws for ${LOTTERY_CONFIGS[lotteryType].displayName}`,
      data: result,
    });
  })
);

/**
 * POST /api/admin/download-all
 * Összes lottó típus frissítése
 */
router.post(
  '/download-all',
  asyncHandler(async (req: Request, res: Response) => {
    const result = await ScraperService.downloadAllLotteries();
    
    const totalNew = result.results.reduce((sum, r) => sum + r.newDraws, 0);
    const totalDraws = result.results.reduce((sum, r) => sum + r.totalDraws, 0);
    
    res.json({
      success: true,
      message: `Downloaded ${totalNew} new draws across all lottery types`,
      summary: {
        totalNewDraws: totalNew,
        totalDrawsInDatabase: totalDraws,
      },
      data: result.results,
    });
  })
);

/**
 * POST /api/admin/init-database
 * Teljes adatbázis inicializálás - minden lottó típus létrehozása és számok letöltése
 */
router.post(
  '/init-database',
  asyncHandler(async (req: Request, res: Response) => {
    const results = [];
    
    for (const configKey of Object.keys(LOTTERY_CONFIGS)) {
      try {
        // Először létrehozzuk a lottó típust
        await ScraperService.ensureLotteryType(configKey);
        
        // Majd letöltjük a számokat
        const result = await ScraperService.downloadAndSaveNumbers(configKey);
        results.push({
          name: LOTTERY_CONFIGS[configKey].displayName,
          status: 'success',
          ...result,
        });
      } catch (error) {
        results.push({
          name: LOTTERY_CONFIGS[configKey].displayName,
          status: 'error',
          error: (error as Error).message,
        });
      }
    }
    
    res.json({
      success: true,
      message: 'Database initialized',
      data: results,
    });
  })
);

/**
 * GET /api/admin/predict/:lotteryType
 * Predikció generálása
 */
router.get(
  '/predict/:lotteryType',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryType } = req.params;
    const { tickets = '4' } = req.query;
    
    const result = await PredictorService.generatePrediction(
      lotteryType,
      parseInt(tickets as string)
    );
    
    res.json({
      success: true,
      data: result,
    });
  })
);

/**
 * GET /api/admin/stats/:lotteryType
 * Statisztikák lekérése
 */
router.get(
  '/stats/:lotteryType',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryType } = req.params;
    
    const stats = await PredictorService.getStats(lotteryType);
    
    res.json({
      success: true,
      data: stats,
    });
  })
);

/**
 * POST /api/admin/reset/:lotteryType
 * Adott lottó típus adatainak törlése és újratöltése
 */
router.post(
  '/reset/:lotteryType',
  asyncHandler(async (req: Request, res: Response) => {
    const { lotteryType } = req.params;
    
    if (!LOTTERY_CONFIGS[lotteryType]) {
      return res.status(400).json({
        success: false,
        error: `Unknown lottery type: ${lotteryType}`,
      });
    }
    
    // Import prisma
    const { prisma } = require('../config/database');
    
    // Lottó típus megkeresése
    const lottery = await prisma.lotteryType.findUnique({
      where: { name: LOTTERY_CONFIGS[lotteryType].name },
    });
    
    if (lottery) {
      // Nyerőszámok törlése
      const deleted = await prisma.winningNumber.deleteMany({
        where: { lotteryTypeId: lottery.id },
      });
      
      // Újratöltés
      const result = await ScraperService.downloadAndSaveNumbers(lotteryType);
      
      res.json({
        success: true,
        message: `Reset complete for ${LOTTERY_CONFIGS[lotteryType].displayName}`,
        data: {
          deletedDraws: deleted.count,
          newDraws: result.newDraws,
          totalDraws: result.totalDraws,
        },
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Lottery type not found in database',
      });
    }
  })
);

/**
 * POST /api/admin/reset-all
 * Összes lottó típus adatainak törlése és újratöltése
 */
router.post(
  '/reset-all',
  asyncHandler(async (req: Request, res: Response) => {
    const { prisma } = require('../config/database');
    const results = [];
    
    for (const configKey of Object.keys(LOTTERY_CONFIGS)) {
      try {
        const lottery = await prisma.lotteryType.findUnique({
          where: { name: LOTTERY_CONFIGS[configKey].name },
        });
        
        if (lottery) {
          const deleted = await prisma.winningNumber.deleteMany({
            where: { lotteryTypeId: lottery.id },
          });
          
          const result = await ScraperService.downloadAndSaveNumbers(configKey);
          
          results.push({
            name: LOTTERY_CONFIGS[configKey].displayName,
            status: 'success',
            deletedDraws: deleted.count,
            newDraws: result.newDraws,
            totalDraws: result.totalDraws,
          });
        }
      } catch (error) {
        results.push({
          name: LOTTERY_CONFIGS[configKey].displayName,
          status: 'error',
          error: (error as Error).message,
        });
      }
    }
    
    res.json({
      success: true,
      message: 'Reset complete for all lottery types',
      data: results,
    });
  })
);

export default router;
