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

            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenerateLottery, 1000, "By Interval for Several Times");
            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenereateRandom, 1000, "By Interval for Several Times");
            StatisticHandler.RunMethodWithEachTimeAndGetTheBestNumbers(StatisticHandler.GenerateNumbersFromSum, 1000, "By Interval for Several Times");
            


            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateFromInterVal, 1, "By Interval");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateLottery, 2, "By Occurrence");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateAvarageStepLines, 1, "By Avarage Steps");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenereateRandom, 2, "By Avarage Randoms");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateNumbersFromSum, 2, "By Sums");
            StatisticHandler.UseEarlierWeekPercentageForNumbersDraw();
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.CalcTheFiveMostCommonNumbers, 1, "By Distribution Based Current Draw");

            //StatisticHandler.SaveCurrentNumbersToFileWithJson("test.json"); 
            Console.ReadKey();
        }
    }

}
