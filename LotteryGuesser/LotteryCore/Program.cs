using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Mail;
using System.Text;
using System.Threading;
using FluentEmail.Core;
using FluentEmail.Mailgun;
using LotteryCore.Model;
using LotteryGuesser.Model;
using OfficeOpenXml;

namespace LotteryCore
{
    class Program
    {
        

        static void Main(string[] args)
        {





            StatisticHandler.DownloadNumbersFromInternet();
            StatisticHandler.GenerateSections();
            

            
           
            

            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateFromInterVal, 1, "By Interval");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateLottery, 2, "By Occurrence");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateAvarageStepLines, 1, "By Avarage Steps");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenereateRandom, 2, "By Avarage Randoms");
            StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateNumbersFromSum, 2, "By Sums");
            StatisticHandler.SaveCurrentNumbersToFileWithJson("test.json");
            Console.ReadKey();
        }

        

        
    }

}
