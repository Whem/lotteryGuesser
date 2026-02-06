/**
 * Lottery Scraper Service
 * Letölti és feldolgozza a lottó számokat a szerencsejatek.hu oldalról
 */

import axios from 'axios';
import * as cheerio from 'cheerio';
import { prisma } from '../config/database';
import { logger } from '../utils/logger';
import { AppError } from '../middleware/errorHandler';

// Előre definiált lottó típusok konfigurációja a pontos húzási időpontokkal
export const LOTTERY_CONFIGS: Record<string, {
  name: string;
  displayName: string;
  url?: string;                // Scraper URL (csak magyar lottóknál)
  minNumber: number;
  maxNumber: number;
  numbersCount: number;        // Hány számot húznak a sorsoláson
  userPicksCount?: number;     // Hány számot tippelhet a felhasználó (ha különbözik a húzottól)
  skipItems?: number;
  hasAdditionalNumbers: boolean;
  additionalMinNumber?: number;
  additionalMaxNumber?: number;
  additionalNumbersCount?: number;
  numbersInSingleCell?: boolean;
  drawDays: string[];
  drawTime: string;
  updateSchedule?: { day: number; hour: number; minute: number }[];
  // Új mezők
  country: string;
  countryCode: string;
  emoji: string;
  playDomain: string;
  timezone: string;
  isInternational?: boolean;   // Ha true, nincs scraper
}> = {
  // ==================== MAGYAR LOTTÓK ====================
  'otos-lotto': {
    name: 'otos-lotto',
    displayName: 'Ötöslottó',
    url: 'https://bet.szerencsejatek.hu/cmsfiles/otos.html',
    minNumber: 1,
    maxNumber: 90,
    numbersCount: 5,
    skipItems: 2,
    hasAdditionalNumbers: false,
    drawDays: ['Saturday'],
    drawTime: '18:45',
    updateSchedule: [{ day: 0, hour: 8, minute: 0 }],
    country: 'Magyarország',
    countryCode: 'HU',
    emoji: '5️⃣',
    playDomain: 'bet.szerencsejatek.hu',
    timezone: 'Europe/Budapest',
  },
  'hatos-lotto': {
    name: 'hatos-lotto',
    displayName: 'Hatoslottó',
    url: 'https://bet.szerencsejatek.hu/cmsfiles/hatos.html',
    minNumber: 1,
    maxNumber: 45,
    numbersCount: 6,
    skipItems: 2,
    hasAdditionalNumbers: false,
    drawDays: ['Thursday', 'Sunday'],
    drawTime: '20:50 / 16:00',
    updateSchedule: [
      { day: 5, hour: 8, minute: 0 },
      { day: 1, hour: 8, minute: 0 },
    ],
    country: 'Magyarország',
    countryCode: 'HU',
    emoji: '6️⃣',
    playDomain: 'bet.szerencsejatek.hu',
    timezone: 'Europe/Budapest',
  },
  'skandinav-lotto': {
    name: 'skandinav-lotto',
    displayName: 'Skandináv lottó',
    url: 'https://bet.szerencsejatek.hu/cmsfiles/skandi.html',
    minNumber: 1,
    maxNumber: 35,
    numbersCount: 7,
    skipItems: 2,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 35,
    additionalNumbersCount: 7,
    drawDays: ['Wednesday'],
    drawTime: '20:50',
    updateSchedule: [{ day: 4, hour: 8, minute: 0 }],
    country: 'Magyarország',
    countryCode: 'HU',
    emoji: '❄️',
    playDomain: 'bet.szerencsejatek.hu',
    timezone: 'Europe/Budapest',
  },
  'eurojackpot': {
    name: 'eurojackpot',
    displayName: 'Eurojackpot',
    url: 'https://bet.szerencsejatek.hu/cmsfiles/eurojackpot.html',
    minNumber: 1,
    maxNumber: 50,
    numbersCount: 5,
    skipItems: 2,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 12,
    additionalNumbersCount: 2,
    numbersInSingleCell: false,
    drawDays: ['Tuesday', 'Friday'],
    drawTime: '20:00-21:00',
    updateSchedule: [
      { day: 3, hour: 8, minute: 0 },
      { day: 6, hour: 8, minute: 0 },
    ],
    country: 'Magyarország',
    countryCode: 'HU',
    emoji: '🎰',
    playDomain: 'bet.szerencsejatek.hu',
    timezone: 'Europe/Budapest',
  },
  'keno': {
    name: 'keno',
    displayName: 'Kenó',
    url: 'https://bet.szerencsejatek.hu/cmsfiles/keno.html',
    minNumber: 1,
    maxNumber: 80,
    numbersCount: 20,
    userPicksCount: 10,
    skipItems: 4,
    hasAdditionalNumbers: false,
    numbersInSingleCell: false,
    drawDays: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    drawTime: 'Naponta 22:00',
    updateSchedule: [
      { day: 0, hour: 8, minute: 0 },
      { day: 1, hour: 8, minute: 0 },
      { day: 2, hour: 8, minute: 0 },
      { day: 3, hour: 8, minute: 0 },
      { day: 4, hour: 8, minute: 0 },
      { day: 5, hour: 8, minute: 0 },
      { day: 6, hour: 8, minute: 0 },
    ],
    country: 'Magyarország',
    countryCode: 'HU',
    emoji: '🎱',
    playDomain: 'bet.szerencsejatek.hu',
    timezone: 'Europe/Budapest',
  },

  // ==================== NEMZETKÖZI LOTTÓK ====================
  'iceland-lotto': {
    name: 'iceland-lotto',
    displayName: 'Lottó',
    minNumber: 1,
    maxNumber: 45,
    numbersCount: 5,
    hasAdditionalNumbers: false,
    drawDays: ['Saturday'],
    drawTime: '18:54 GMT',
    country: 'Izland',
    countryCode: 'IS',
    emoji: '🧊',
    playDomain: 'games.lotto.is',
    timezone: 'Atlantic/Reykjavik',
    isInternational: true,
  },
  'norway-lotto': {
    name: 'norway-lotto',
    displayName: 'Lotto',
    minNumber: 1,
    maxNumber: 34,
    numbersCount: 7,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 34,
    additionalNumbersCount: 1,
    drawDays: ['Saturday'],
    drawTime: '19:45',
    country: 'Norvégia',
    countryCode: 'NO',
    emoji: '🍀',
    playDomain: 'norsk-tipping.no',
    timezone: 'Europe/Oslo',
    isInternational: true,
  },
  'swiss-lotto': {
    name: 'swiss-lotto',
    displayName: 'Swiss Lotto',
    minNumber: 1,
    maxNumber: 42,
    numbersCount: 6,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 6,
    additionalNumbersCount: 1,
    drawDays: ['Wednesday', 'Saturday'],
    drawTime: '19:00',
    country: 'Svájc',
    countryCode: 'CH',
    emoji: '🎟️',
    playDomain: 'swisslos.ch',
    timezone: 'Europe/Zurich',
    isInternational: true,
  },
  'denmark-lotto': {
    name: 'denmark-lotto',
    displayName: 'Lotto',
    minNumber: 1,
    maxNumber: 36,
    numbersCount: 7,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 36,
    additionalNumbersCount: 1,
    drawDays: ['Saturday'],
    drawTime: '21:00',
    country: 'Dánia',
    countryCode: 'DK',
    emoji: '🟦',
    playDomain: 'danskespil.dk',
    timezone: 'Europe/Copenhagen',
    isInternational: true,
  },
  'germany-lotto': {
    name: 'germany-lotto',
    displayName: 'LOTTO 6aus49',
    minNumber: 1,
    maxNumber: 49,
    numbersCount: 6,
    hasAdditionalNumbers: true,
    additionalMinNumber: 0,
    additionalMaxNumber: 9,
    additionalNumbersCount: 1,
    drawDays: ['Wednesday', 'Saturday'],
    drawTime: '18:25 / 19:25',
    country: 'Németország',
    countryCode: 'DE',
    emoji: '🧮',
    playDomain: 'lotto.de',
    timezone: 'Europe/Berlin',
    isInternational: true,
  },
  'sweden-lotto': {
    name: 'sweden-lotto',
    displayName: 'Lotto',
    minNumber: 1,
    maxNumber: 35,
    numbersCount: 7,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 35,
    additionalNumbersCount: 7,
    drawDays: ['Wednesday', 'Saturday'],
    drawTime: '19:30',
    country: 'Svédország',
    countryCode: 'SE',
    emoji: '⭐',
    playDomain: 'svenskaspel.se',
    timezone: 'Europe/Stockholm',
    isInternational: true,
  },
  'australia-ozlotto': {
    name: 'australia-ozlotto',
    displayName: 'Oz Lotto',
    minNumber: 1,
    maxNumber: 47,
    numbersCount: 7,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 47,
    additionalNumbersCount: 3,
    drawDays: ['Tuesday'],
    drawTime: '19:30 AEST',
    country: 'Ausztrália',
    countryCode: 'AU',
    emoji: '🦘',
    playDomain: 'thelott.com',
    timezone: 'Australia/Sydney',
    isInternational: true,
  },
  'netherlands-lotto': {
    name: 'netherlands-lotto',
    displayName: 'Lotto',
    minNumber: 1,
    maxNumber: 45,
    numbersCount: 6,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 45,
    additionalNumbersCount: 1,
    drawDays: ['Saturday'],
    drawTime: '21:00',
    country: 'Hollandia',
    countryCode: 'NL',
    emoji: '🎯',
    playDomain: 'lotto.nederlandseloterij.nl',
    timezone: 'Europe/Amsterdam',
    isInternational: true,
  },
  'hongkong-marksix': {
    name: 'hongkong-marksix',
    displayName: 'Mark Six',
    minNumber: 1,
    maxNumber: 49,
    numbersCount: 6,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 49,
    additionalNumbersCount: 1,
    drawDays: ['Tuesday', 'Thursday', 'Saturday'],
    drawTime: '21:30 HKT',
    country: 'Hongkong',
    countryCode: 'HK',
    emoji: '🐎',
    playDomain: 'bet.hkjc.com',
    timezone: 'Asia/Hong_Kong',
    isInternational: true,
  },
  'belgium-euromillions': {
    name: 'belgium-euromillions',
    displayName: 'EuroMillions',
    minNumber: 1,
    maxNumber: 50,
    numbersCount: 5,
    hasAdditionalNumbers: true,
    additionalMinNumber: 1,
    additionalMaxNumber: 12,
    additionalNumbersCount: 2,
    drawDays: ['Tuesday', 'Friday'],
    drawTime: '21:30',
    country: 'Belgium',
    countryCode: 'BE',
    emoji: '💶',
    playDomain: 'loterie-nationale.be',
    timezone: 'Europe/Brussels',
    isInternational: true,
  },
};

export interface ScrapedDraw {
  year: number;
  week: number;
  numbers: number[];
  additionalNumbers?: number[];
}

export class ScraperService {
  /**
   * Lottó számok letöltése és feldolgozása egy URL-ről
   */
  static async scrapeFromUrl(
    url: string,
    numbersCount: number,
    skipItems: number,
    hasAdditionalNumbers: boolean,
    additionalNumbersCount: number = 0,
    numbersInSingleCell: boolean = false,
    maxNumber: number = 90
  ): Promise<ScrapedDraw[]> {
    try {
      logger.info(`Scraping lottery data from: ${url} (singleCell: ${numbersInSingleCell})`);
      
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'hu-HU,hu;q=0.9,en;q=0.8',
        },
        timeout: 30000,
      });

      const html = response.data;
      const $ = cheerio.load(html);
      
      const draws: ScrapedDraw[] = [];
      const rows = $('table tr').slice(1); // Skip header
      
      rows.each((_, row) => {
        const cols = $(row).find('td');
        
        if (cols.length >= 3) { // Minimum: év, hét, + legalább 1 szám oszlop
          try {
            const year = parseInt($(cols.get(0)).text().trim());
            const week = parseInt($(cols.get(1)).text().trim());
            
            if (isNaN(year) || isNaN(week) || year < 1900 || year > 2100) return;
            
            let numbers: number[] = [];
            let additionalNumbers: number[] | undefined;
            
            if (numbersInSingleCell) {
              // EUROJACKPOT stílus: Számok egyetlen cellában vannak szóközzel elválasztva
              // Az utolsó oszlop tartalmazza a számokat, pl: "8 16 37 39 48 5 11"
              const lastColText = $(cols.get(cols.length - 1)).text().trim();
              const allNumbers = lastColText.split(/\s+/).map(n => parseInt(n)).filter(n => !isNaN(n) && n >= 1);
              
              if (allNumbers.length >= numbersCount) {
                // Első `numbersCount` szám a fő számok
                numbers = allNumbers.slice(0, numbersCount);
                
                // Maradék a kiegészítő számok
                if (hasAdditionalNumbers && allNumbers.length > numbersCount) {
                  additionalNumbers = allNumbers.slice(numbersCount);
                }
              }
            } else {
              // NORMÁL stílus: Számok külön oszlopokban vannak az utolsó oszlopokban
              const totalNumbers = numbersCount + (hasAdditionalNumbers ? additionalNumbersCount : 0);
              const numbersStartIdx = cols.length - totalNumbers;
              
              // Fő számok
              for (let i = numbersStartIdx; i < numbersStartIdx + numbersCount && i < cols.length; i++) {
                const text = $(cols.get(i)).text().trim().replace(/\s/g, '');
                const num = parseInt(text);
                if (!isNaN(num) && num >= 1 && num <= maxNumber) numbers.push(num);
              }
              
              // Kiegészítő számok
              if (hasAdditionalNumbers && additionalNumbersCount > 0) {
                additionalNumbers = [];
                for (let i = numbersStartIdx + numbersCount; i < cols.length; i++) {
                  const text = $(cols.get(i)).text().trim().replace(/\s/g, '');
                  const num = parseInt(text);
                  if (!isNaN(num) && num >= 1 && num <= maxNumber) additionalNumbers.push(num);
                }
                if (additionalNumbers.length === 0) additionalNumbers = undefined;
              }
            }
            
            if (numbers.length !== numbersCount) return;
            
            draws.push({ year, week, numbers, additionalNumbers });
          } catch (e) {
            // Skip malformed rows
          }
        }
      });
      
      // Fordított sorrend (legrégebbi elől)
      draws.reverse();
      
      logger.info(`Scraped ${draws.length} draws from ${url}`);
      return draws;
      
    } catch (error) {
      logger.error(`Error scraping ${url}:`, error);
      throw new AppError(`Failed to scrape lottery data: ${(error as Error).message}`, 500, 'SCRAPE_ERROR');
    }
  }

  /**
   * Lottó típus adatbázisba mentése vagy frissítése
   */
  static async ensureLotteryType(configKey: string) {
    const config = LOTTERY_CONFIGS[configKey];
    if (!config) {
      throw new AppError(`Unknown lottery type: ${configKey}`, 400, 'UNKNOWN_LOTTERY_TYPE');
    }

    // Meglévő keresése vagy létrehozása
    let lottery = await prisma.lotteryType.findUnique({
      where: { name: config.name },
    });

    if (!lottery) {
      lottery = await prisma.lotteryType.create({
        data: {
          name: config.name,
          displayName: config.displayName,
          country: 'Hungary',
          description: `Magyar ${config.displayName}`,
          minNumber: config.minNumber,
          maxNumber: config.maxNumber,
          numbersCount: config.numbersCount,
          hasAdditionalNumbers: config.hasAdditionalNumbers,
          additionalMinNumber: config.additionalMinNumber,
          additionalMaxNumber: config.additionalMaxNumber,
          additionalNumbersCount: config.additionalNumbersCount,
          drawDays: config.drawDays,
          currency: 'HUF',
        },
      });
      logger.info(`Created lottery type: ${config.displayName}`);
    }

    return lottery;
  }

  /**
   * Lottó számok letöltése és mentése az adatbázisba
   */
  static async downloadAndSaveNumbers(configKey: string): Promise<{
    lotteryId: string;
    newDraws: number;
    totalDraws: number;
    skipped?: boolean;
  }> {
    const config = LOTTERY_CONFIGS[configKey];
    if (!config) {
      throw new AppError(`Unknown lottery type: ${configKey}`, 400, 'UNKNOWN_LOTTERY_TYPE');
    }

    // Biztosítjuk, hogy létezik a lottó típus
    const lottery = await this.ensureLotteryType(configKey);

    // Nemzetközi lottóknál nincs scraper - csak az adatbázis típust hozzuk létre
    if (config.isInternational || !config.url) {
      const totalDraws = await prisma.winningNumber.count({
        where: { lotteryTypeId: lottery.id },
      });
      return {
        lotteryId: lottery.id,
        newDraws: 0,
        totalDraws,
        skipped: true,
      };
    }

    // Letöltjük a számokat
    const draws = await this.scrapeFromUrl(
      config.url,
      config.numbersCount,
      config.skipItems || 0,
      config.hasAdditionalNumbers,
      config.additionalNumbersCount || 0,
      config.numbersInSingleCell || false,
      config.maxNumber
    );

    let newDraws = 0;

    for (const draw of draws) {
      // Ellenőrizzük, hogy létezik-e már
      const existing = await prisma.winningNumber.findUnique({
        where: {
          lotteryTypeId_drawYear_drawWeek: {
            lotteryTypeId: lottery.id,
            drawYear: draw.year,
            drawWeek: draw.week,
          },
        },
      });

      if (!existing) {
        // Dátum számítása év és hét alapján
        const drawDate = this.getDateFromYearWeek(draw.year, draw.week);
        
        await prisma.winningNumber.create({
          data: {
            lotteryTypeId: lottery.id,
            numbers: draw.numbers.sort((a, b) => a - b),
            additionalNumbers: draw.additionalNumbers?.sort((a, b) => a - b),
            drawDate,
            drawYear: draw.year,
            drawWeek: draw.week,
          },
        });
        newDraws++;
      }
    }

    const totalDraws = await prisma.winningNumber.count({
      where: { lotteryTypeId: lottery.id },
    });

    logger.info(`Downloaded ${newDraws} new draws for ${config.displayName}. Total: ${totalDraws}`);

    return {
      lotteryId: lottery.id,
      newDraws,
      totalDraws,
    };
  }

  /**
   * Összes előre definiált lottó típus frissítése
   */
  static async downloadAllLotteries(): Promise<{
    results: Array<{
      name: string;
      lotteryId: string;
      newDraws: number;
      totalDraws: number;
      error?: string;
    }>;
  }> {
    const results = [];

    for (const configKey of Object.keys(LOTTERY_CONFIGS)) {
      try {
        const result = await this.downloadAndSaveNumbers(configKey);
        results.push({
          name: LOTTERY_CONFIGS[configKey].displayName,
          ...result,
        });
      } catch (error) {
        results.push({
          name: LOTTERY_CONFIGS[configKey].displayName,
          lotteryId: '',
          newDraws: 0,
          totalDraws: 0,
          error: (error as Error).message,
        });
      }
    }

    return { results };
  }

  /**
   * Év és hét alapján dátum számítása
   */
  private static getDateFromYearWeek(year: number, week: number): Date {
    // ISO hét alapján számítjuk a dátumot
    const simple = new Date(year, 0, 1 + (week - 1) * 7);
    const dayOfWeek = simple.getDay();
    const ISOweekStart = simple;
    
    if (dayOfWeek <= 4) {
      ISOweekStart.setDate(simple.getDate() - simple.getDay() + 1);
    } else {
      ISOweekStart.setDate(simple.getDate() + 8 - simple.getDay());
    }
    
    // Szombatra állítjuk (a legtöbb húzás szombaton van)
    ISOweekStart.setDate(ISOweekStart.getDate() + 5);
    
    return ISOweekStart;
  }

  /**
   * Lottó típusok listázása
   */
  static getAvailableLotteryTypes() {
    return Object.entries(LOTTERY_CONFIGS).map(([key, config]) => ({
      key,
      name: config.name,
      displayName: config.displayName,
      url: config.url,
      minNumber: config.minNumber,
      maxNumber: config.maxNumber,
      numbersCount: config.userPicksCount || config.numbersCount, // Hány számot tippel a felhasználó
      drawnCount: config.numbersCount, // Hány számot húznak
      hasAdditionalNumbers: config.hasAdditionalNumbers,
      additionalMaxNumber: config.additionalMaxNumber,
      additionalNumbersCount: config.additionalNumbersCount,
      drawDays: config.drawDays,
      drawTime: config.drawTime,
      // Új mezők
      country: config.country,
      countryCode: config.countryCode,
      emoji: config.emoji,
      playDomain: config.playDomain,
      timezone: config.timezone,
      isInternational: config.isInternational || false,
    }));
  }
}
