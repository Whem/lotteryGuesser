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
using LotteryCore.Tools;
using LotteryGuesser.Model;
using OfficeOpenXml;

namespace LotteryCore
{
    class Program
    {

        
        static void Main(string[] args)
        {
            var gsd = new GoogleSheetData();
            var lh = new LotteryHandler();

            lh.DownloadNumbersFromInternet("https://bet.szerencsejatek.hu/cmsfiles/otos.html");
            lh.GenerateSections();
            lh.LoadNumbersFromSheet(gsd.GetData());

            lh.MakeStatisticFromEarlierWeek();

            lh.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach,2);
            lh.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);

           
            lh.UseEarlierWeekPercentageForNumbersDraw( Enums.TypesOfDrawn.Calculated );
            lh.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique,1);

            //gsd.SaveNumbersToSheet(lh.LotteryModels);
            Console.ReadKey();
        }
    }

}
