using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Mail;
using System.Text;
using System.Threading;
using FluentEmail.Core;
using FluentEmail.Mailgun;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Services;
using Google.Apis.Sheets.v4;
using Google.Apis.Sheets.v4.Data;
using Google.Apis.Util.Store;
using LotteryCore.Model;
using LotteryGuesser.Model;
using OfficeOpenXml;

namespace LotteryCore
{
    class Program
    {

        
        static void Main(string[] args)
        {
            var gsd = new GoogleSheetData();


            StatisticHandler.DownloadNumbersFromInternet("https://bet.szerencsejatek.hu/cmsfiles/otos.html");
            StatisticHandler.GenerateSections();
            StatisticHandler.LoadNumbersFromSheet(gsd.GetData());
            
            StatisticHandler.MakeStatisticFromEarlierWeek();

            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenerateLottery, 1000, SaveNumber.TypesOfDrawn.ByIntervalForEachTimes);
            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenerateRandom, 1000, SaveNumber.TypesOfDrawn.ByIntervalForEachTimes);
            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenerateNumbersFromSum, 1000, SaveNumber.TypesOfDrawn.ByIntervalForEachTimes);
            


            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateFromInterVal, 1, SaveNumber.TypesOfDrawn.ByInterval);
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateLottery, 2, SaveNumber.TypesOfDrawn.ByOccurrence);
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateAverageStepLines, 1,SaveNumber.TypesOfDrawn.ByAverageSteps);
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateRandom, 2, SaveNumber.TypesOfDrawn.ByAverageRandoms);
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateNumbersFromSum, 2, SaveNumber.TypesOfDrawn.BySums);
            StatisticHandler.UseEarlierWeekPercentageForNumbersDraw( SaveNumber.TypesOfDrawn.Calculated );
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.CalcTheFiveMostCommonNumbers, 1, SaveNumber.TypesOfDrawn.ByDistributionBasedCurrentDraw);

            //gsd.SaveNumbersToSheet();
            Console.ReadKey();
        }
    }

}
