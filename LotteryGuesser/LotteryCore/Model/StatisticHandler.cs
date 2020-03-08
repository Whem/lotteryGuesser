using LotteryGuesser.Model;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using LotteryCore.Tools;

namespace LotteryCore.Model
{
    public static class StatisticHandler
    {
        public const string UserName = "Whem";

        public static LotteryStatistic LotteryStatistic;

        public static List<SaveNumber> SaveNumbers = new List<SaveNumber>();

        static List<LotteryModel> lotteryCollection;

        static List<NumberSections> numberSections = new List<NumberSections>();
        private static List<LotteryModel> _lotteryModels;

        public static List<LotteryModel> LotteryModels
        {
            get => _lotteryModels;
            set => _lotteryModels = value;
        }


        public static void DownloadNumbersFromInternet(string link)
        {
            lotteryCollection = new List<LotteryModel>();
            using (WebClient client = new WebClient()) // WebClient class inherits IDisposable
            {
                // Or you can get the file content without saving it
                string htmlCode = client.DownloadString(link);
                HtmlAgilityPack.HtmlDocument doc = new HtmlAgilityPack.HtmlDocument();
                doc.LoadHtml(htmlCode);

                List<List<string>> table = doc.DocumentNode.SelectSingleNode("//table")
                            .Descendants("tr")
                            .Skip(1)
                            .Where(tr => tr.Elements("td").Count() > 1)
                            .Select(tr => tr.Elements("td").Select(td => td.InnerText.Trim()).ToList())
                            .ToList();
                int index = 0;
                table.Reverse();
                foreach (List<string> list in table)
                {
                    lotteryCollection.Add(new LotteryModel(list, index));
                    index++;
                }
                
            }
        }

        internal static void LoadNumbersFromSheet(List<string[]> getData)
        {
            SaveNumbers = new List<SaveNumber>();
            if (getData != null && getData.Count > 0)
            {

                foreach (var row in getData)
                {
                    SaveNumbers.Add(new SaveNumber(row));
                }
            }
            else
            {
                Console.WriteLine("No data found.");
            }
        }

        public static List<LotteryModel> GetLotteryCollection()
        {
            return lotteryCollection;
        }

        public static List<NumberSections> GetLotteryStatistics()
        {
            return numberSections;
        }

        public static void AddNumbersToSaveFile(SaveNumber saveNumber)
        {
            SaveNumbers.Add(saveNumber);
        }        

        public static void GenerateSections()
        {
            LotteryStatistic = new LotteryStatistic(lotteryCollection);


            for (int i = 1; i < 91; i++)
            {
                var actualNumberSection = new NumberSections(i);
                int loti = 1;
                foreach (var t in lotteryCollection)
                {
                    int index = t.Numbers.FindIndex(a => a == i);
                    if (index < 5 && index >= 0 && loti < lotteryCollection.Count)
                    {
                        actualNumberSection.FindTheNextNumber(lotteryCollection[loti].Numbers[index]);

                        var getInterval = LotteryStatistic.IntervallNumbers.First(x => x.StartInterVal <= lotteryCollection[loti].Numbers[index] && x.StopInterval >= lotteryCollection[loti].Numbers[index]);
                        getInterval.ActualNumberList.Add(t.Numbers[index]);
                        getInterval.AfterNumberList.Add(lotteryCollection[loti].Numbers[index]);
                    }

                    loti++;

                }
                numberSections.Add(actualNumberSection);
            }
        }

        public static void RunMethodWithEachTime(Func<LotteryModel> action, int count, SaveNumber.TypesOfDrawn tDrawn)
        {
            if(LotteryModels == null)LotteryModels = new List<LotteryModel>();
            Console.WriteLine(tDrawn.ToString());
            int index = 0;
            while (true)
            {
                LotteryModel returnedModel= action();

                if(returnedModel == null) continue;
                if (LotteryModels.AddValueWithDetailsAndValidation(returnedModel.ValidationTuple(),tDrawn))
                {
                    index++;
                }
                else
                {
                    continue;
                }
                if (index == count)
                {
                    break;
                }

            }
        }

        public static void RunMethodWithEachTimeAndGetTheBestNumbers(Func<LotteryModel> action, int count, SaveNumber.TypesOfDrawn tDrawn)
        {
            Dictionary<int,int> numbersDictionary = new Dictionary<int, int>();
            
            int index = 0;
            while (index != count)
            {
                var returnedModel = action();


                if (returnedModel == null || !returnedModel.ValidationTuple().Item1) continue;

                index++;
                foreach (var returnedModelNumber in returnedModel.Numbers)
                {
                    if (numbersDictionary.Count > 0 && numbersDictionary.ContainsKey(returnedModelNumber))
                    {
                        numbersDictionary[returnedModelNumber]++;
                    }
                    else
                    {
                        numbersDictionary.Add(returnedModelNumber, 1);
                    }
                }
            }

            var sortedDics = numbersDictionary.OrderByDescending(x => x.Value).Take(5);
            LotteryModel resultLotteryModel = new LotteryModel();
            foreach (KeyValuePair<int, int> keyValuePair in sortedDics)
            {
                resultLotteryModel.Numbers.Add(keyValuePair.Key);
            }

            LotteryModels.AddValueWithDetailsAndValidation(resultLotteryModel.ValidationTuple(),tDrawn);
        }

        public static LotteryModel CalcTheFiveMostCommonNumbers()
        {
            return new LotteryModel()
            {
                Numbers = LotteryModels.SelectMany(x => x.Numbers).ToList().GroupBy(n => n)
                .Select(n => new
                    {
                        MetricName = n.Key,
                        MetricCount = n.Count()
                    }
                )
                .OrderByDescending(n => n.MetricCount).Take(5).Select(x => x.MetricName).ToList()
            };
        }
        public static LotteryModel GenerateFromInterVal()
        {
            
                var acutal = numberSections.Where(x => x.ActualNumber == lotteryCollection.Last().FirstNumber);

                LotteryModel lm = new LotteryModel();
                for (int i = 0; i < 5; i++)
                {
                    int maxcoount = 0;
                    List<IOrderedEnumerable<SpecifyNumber>> spec = acutal.Select(x => x.SpecifyNumberList.OrderByDescending(xy => xy.Pieces)).ToList();
                    List<int> maxCount = spec[0].Select(x => x.Pieces).OrderByDescending(x => x).ToList();
                    SpecifyNumber getsec = spec[0].First(x => x.Pieces == maxCount[maxcoount]);

                    while (true)
                    {
                        if (lm.Numbers.Contains(getsec.NextNumber))
                        {
                            maxcoount++;
                            getsec = spec[0].First(x => x.Pieces == maxCount[maxcoount]);
                        }
                        else
                        {
                            if (lm.Numbers.Count == 0)
                                break;
                            if (getsec.NextNumber > lm.Numbers.Max())
                                break;
                            maxcoount++;
                            getsec = spec[0].First(x => x.Pieces == maxCount[maxcoount]);
                        }
                    }

                    var getAf = getsec.NextNumber;
                    lm.Numbers.Add(getAf);
                    acutal = numberSections.Where(x => x.ActualNumber == getAf);

                }

                return lm;



        }

        public static LotteryModel GenerateNumbersFromSum()
        {
            var getLastSum = lotteryCollection.Last().Sum;
            LotteryModel getBeforeLastSumId;
            while (true)
            {
                Random rnd = new Random();
                var foundLastSum= lotteryCollection
                    .Select((v, i) => new { Index = i, Value = v })
                    .Where(x => x.Value.Sum == getLastSum)
                    .Select(x => x.Value.Id)
                    .ToList();
                if (foundLastSum.Count > 2)
                {
                    var tek = rnd.Next(1, foundLastSum.Count - 1);
                    getBeforeLastSumId = lotteryCollection.FirstOrDefault(x => x.Id == foundLastSum[tek] + 1);
                   
                    break;
                }

                getLastSum++;
            }


            
            LotteryModel lm = new LotteryModel();
            Random rnd2 = new Random();
            for (int i = 0; i < 5; i++)
            {
                lm.AddNumber(rnd2.Next(1, 91));
            }
            
            if (lm.Sum >= getBeforeLastSumId.Sum - 10 && lm.Sum <= getBeforeLastSumId.Sum + 10)
            {
                return lm;
            }

            return null;
        }

        public static LotteryModel GenerateRandom()
        {
                LotteryModel lm = new LotteryModel();
                for (int i = 0; i < 5; i++)
                {
                    var goal = LotteryStatistic.AvarageRandom[i];
                    int id = 0;
                    while (true)
                    {
                        Random rnd = new Random();
                        var number = rnd.Next(1, 91);
                        if (id == (int)goal)
                        {
                            lm.AddNumber(number);
                            break;
                        }
                        id++;
                    }
                }

                return lm;
        }

        public static LotteryModel GenerateAverageStepLines()
        {
           
                Random random = new Random();
                int start2 = random.Next(1, 90);
                LotteryModel lm = new LotteryModel();
                lm.AddNumber(start2);

                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage1to2, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage2to3, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage3to4, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage4to5, 0));
                
                
                return lm;

        }


        public static LotteryModel GenerateLottery()
        {
                LotteryModel lm = new LotteryModel();



                var getSections = numberSections.Last(x => lotteryCollection.Last().FirstNumber == x.ActualNumber);
                Random random = new Random();
                SpecifyNumber generateNumber = null;
                int start2 = random.Next(0, getSections.SpecifyNumberList.Count);
                generateNumber = (SpecifyNumber)getSections.SpecifyNumberList[start2].Clone();
                lm.AddNumber(generateNumber.NextNumber);
                for (int k = 0; k < 4; k++)
                {

                    getSections = numberSections.First(x => lm.Numbers.Last() == x.ActualNumber);


                    while (true)
                    {
                        random = new Random();
                        if (getSections.SpecifyNumberList.Count > 10)
                        {
                            start2 = random.Next(0, getSections.SpecifyNumberList.Count);
                            generateNumber = (SpecifyNumber)getSections.SpecifyNumberList[start2].Clone();
                        }
                        else
                        {
                            start2 = random.Next(0, 90);
                            generateNumber = new SpecifyNumber(start2);
                        }




                        if (!lm.Numbers.Contains(generateNumber.NextNumber))
                        {
                            break;
                        }
                    }

                    lm.AddNumber(generateNumber.NextNumber);
                }
                return lm;

        }

        public static void MakeStatisticFromEarlierWeek()
        {
            var getLotteryDrawing =  SaveNumbers.Where(x=> x.WeekOfPull ==lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            var lastDrawning = lotteryCollection.Last();
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < getLotteryDrawing.Count(); j++)
                {
                    var e = getLotteryDrawing[j].Numbers[i];
                    var k = lastDrawning.Numbers[i];
                    var t = (double) e /k ;
                    getLotteryDrawing[j].DifferentInPercentage.Add(t);
                }
            }
        }

       
        public static void UseEarlierWeekPercentageForNumbersDraw(SaveNumber.TypesOfDrawn tDrawn)
        {
            List<SaveNumber> getLotteryDrawing = SaveNumbers.Where(x => x.WeekOfPull == lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            if (getLotteryDrawing.Count == 0)
            {
                Console.WriteLine("You haven't earlier week result");
                return;
            }



            Console.WriteLine("Calculated From earlier week");
            var lmt = LotteryModels.Clone();
           
            foreach (LotteryModel lotteryModel in lmt)
            {
                
                foreach (SaveNumber saveNumber in getLotteryDrawing)
                {
                    LotteryModel lm = new LotteryModel();
                    if (saveNumber.Message != tDrawn) continue;
                    for (int i = 0; i < 5; i++)
                    {
                        double calculatedNumber =
                            lotteryModel.Numbers[i] * saveNumber.DifferentInPercentage[i];
                        lm.Numbers.Add((int)calculatedNumber);
                         
                       
                    }
                    LotteryModels.AddValueWithDetails(lm, tDrawn);
                }
            }

            //int index = 0;
            //int end = LotteryModels.Count;
            //int errCounter = 0;
            //while (true)
            //{
            //    LotteryModel lm = new LotteryModel();
            //    lm.Numbers = new List<int>();
            //    lm.Message = tDrawn;
            //    for (int j = 0; j < 5; j++)
            //    {
            //        var rand = new Random();
            //        double calculatedNumber =
            //            LotteryModels[index].Numbers[j] * getLotteryDrawing[rand.Next(0, getLotteryDrawing.Count - 1)].DifferentInPercentage[j];
            //        lm.Numbers.Add((int)calculatedNumber);
            //    }

            //    if (errCounter == 100)
            //    {
            //        index++;
            //        errCounter = 0;
            //    }

            //    if (IsValidLotteryNumbers(lm))
            //    {
            //        index++;
            //        errCounter = 0;
            //        LotteryModels.Add(lm);
            //        Console.WriteLine(lm);
            //    }
            //    else
            //    {
            //        errCounter++;
            //    }

            //    if (index == end)
            //    {
            //        break;
            //    }
            //}
        }

        // From https://en.wikipedia.org/wiki/ISO_week_date#Weeks_per_year:
        //
        // The long years, with 53 weeks in them, can be described by any of the following equivalent definitions:
        //
        // - Any year starting on Thursday and any leap year starting on Wednesday.
        // - Any year ending on Thursday and any leap year ending on Friday.
        // - Years in which 1 January and 31 December (in common years) or either (in leap years) are Thursdays.
        //
        // All other week-numbering years are short years and have 52 weeks.

        public static int GetWeeksInYear()
        {
            DateTime inputDate = DateTime.Now;
            var d = inputDate;

            CultureInfo cul = CultureInfo.CurrentCulture;
            int weekNum = cul.Calendar.GetWeekOfYear(
                d,
                CalendarWeekRule.FirstDay,
                DayOfWeek.Monday);


            return weekNum;
        }
    }
}
