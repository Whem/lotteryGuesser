using LotteryGuesser.Model;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;

namespace LotteryCore.Model
{
    public static class StatisticHandler
    {
        public static LotteryStatistic LotteryStatistic;

        public static List<SaveNumber> SaveNumbers = new List<SaveNumber>();

        static List<LotteryModel> lotteryCollection;

        static List<NumberSections> numberSections = new List<NumberSections>();

        public static List<LotteryModel> LotteryModels { get; set; }

        public static void DownloadNumbersFromInternet()
        {
            lotteryCollection = new List<LotteryModel>();
            using (WebClient clientt = new WebClient()) // WebClient class inherits IDisposable
            {



                // Or you can get the file content without saving it
                string htmlCode = clientt.DownloadString("https://bet.szerencsejatek.hu/cmsfiles/otos.html");
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

        public static void AddNumbersToSaveFile(SaveNumber saveNumber)
        {
            SaveNumbers.Add(saveNumber);
        }

        public static void LoadNumbersFromJson(string path)
        {
            SaveNumbers =JsonConvert.DeserializeObject<List<SaveNumber>>(File.ReadAllText(path));
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

        public static void RunMethodWithEachTime(Action action, int count, string message)
        {
            LotteryModels = new List<LotteryModel>();
            Console.WriteLine(message);
            for (int i=0; i < count; i++)
            {
                action();
                if(LotteryModels.Count >0)
                SaveNumbers.Add(new SaveNumber(LotteryModels.Last().Numbers.OrderBy(x => x).ToArray(), message));
            }        
            
        }
        
        public static void SaveCurrentNumbersToFileWithJson(string filePath)
        {
            string json = JsonConvert.SerializeObject(SaveNumbers, Formatting.Indented);
            using (var writer = File.CreateText(filePath))
            {
                writer.WriteLine(json); 
            }
        }

        public static void GenerateFromInterVal()
        {
            while (true)
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
                            else
                            {
                                maxcoount++;
                                getsec = spec[0].First(x => x.Pieces == maxCount[maxcoount]);
                            }
                        }
                    }

                    var getAf = getsec.NextNumber;
                    lm.Numbers.Add(getAf);
                    acutal = numberSections.Where(x => x.ActualNumber == getAf);

                }
                var duplicateKeys = lm.Numbers.GroupBy(x => x)
                    .Where(group => group.Count() > 1)
                    .Select(group => group.Key);

                if (!duplicateKeys.Any())
                {
                    LotteryModels.Add(lm);
                    Console.WriteLine(lm);
                    break;
                }

            }


        }

        public static void GenerateNumbersFromSum()
        {
            var getLastSum = lotteryCollection.Last().Sum;
            LotteryModel getbeforeLastSumId;
            while (true)
            {
                Random rnd = new Random();
                var foundHelloWorld = lotteryCollection
                    .Select((v, i) => new { Index = i, Value = v })
                    .Where(x => x.Value.Sum == getLastSum)
                    .Select(x => x.Value.Id)
                    .ToList();
                if (foundHelloWorld.Count > 2)
                {
                    var tek = rnd.Next(1, foundHelloWorld.Count - 1);
                    getbeforeLastSumId = lotteryCollection.FirstOrDefault(x => x.Id == foundHelloWorld[tek] + 1);
                    if (getbeforeLastSumId == null)
                    {

                    }
                    break;
                }

                getLastSum++;


            }


            while (true)
            {
                LotteryModel lm = new LotteryModel();
                Random rnd = new Random();
                for (int i = 0; i < 5; i++)
                {
                    lm.AddNumber(rnd.Next(1, 91));
                }
                if (lotteryCollection.Any(x => x.Numbers.SequenceEqual(lm.Numbers))) continue;
                if (lm.Sum >= getbeforeLastSumId.Sum - 10 && lm.Sum <= getbeforeLastSumId.Sum + 10)
                {
                    var duplicateKeys = lm.Numbers.GroupBy(x => x)
                        .Where(group => group.Count() > 1)
                        .Select(group => group.Key);

                    if (!duplicateKeys.Any())
                    {
                        LotteryModels.Add(lm);
                        Console.WriteLine(lm);
                        break;
                    }
                }
            }
        }

        public static void GenereateRandom()
        {

            while (true)
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

                if (lotteryCollection.Any(x => x.Numbers.SequenceEqual(lm.Numbers)))
                {
                    continue;
                }

                if (lm.Sum >= 440 || lm.Numbers.Max() > 15)
                {
                    continue;
                }
                LotteryModels.Add(lm);
                Console.WriteLine(lm);
                break;
            }


        }

        public static void GenerateAvarageStepLines()
        {
            while (true)
            {
                Random random = new Random();
                int start2 = random.Next(1, 90);
                LotteryModel lm = new LotteryModel();
                lm.AddNumber(start2);

                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage1to2, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage2to3, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage3to4, 0));
                lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(LotteryStatistic.Avarage4to5, 0));


                if (lotteryCollection.Any(x => x.Numbers.SequenceEqual(lm.Numbers)))
                {
                    continue;
                }

                if (lm.Sum >= 440 || lm.Numbers.Max() > 90)
                {
                    continue;
                }
                LotteryModels.Add(lm);
                Console.WriteLine(lm);
                break;
            }
        }


        public static void GenerateLottery()
        {
            while (true)
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


                if (lotteryCollection.Any(x => x.Numbers.SequenceEqual(lm.Numbers))) continue;

                LotteryModels.Add(lm);
                Console.WriteLine(lm);
                break;
            }
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

        public static void UseEarlierWeekPercentageForNumbersDraw()
        {
            var getLotteryDrawing = SaveNumbers.Where(x => x.WeekOfPull == lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            var actualWeekNumbers = SaveNumbers.Where(x => x.WeekOfPull == GetWeeksInYear()).ToList();
            Console.WriteLine("Calculted From earlier week");
            for (int i = 0; i < actualWeekNumbers.Count; i++)
            {
                SaveNumber saveNumber = new SaveNumber();
                saveNumber.Numbers= new List<int>();
                saveNumber.Message = "Calculated";
                for (int j = 0; j < 5; j++)
                {
                    var rand = new Random();
                    double calculatedNumber =
                        actualWeekNumbers[i].Numbers[j] * getLotteryDrawing[rand.Next(0,getLotteryDrawing.Count-1)].DifferentInPercentage[j];
                    saveNumber.Numbers.Add((int)calculatedNumber); 
                }
                SaveNumbers.Add(saveNumber);
                Console.WriteLine(saveNumber);
            }
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
