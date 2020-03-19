using LotteryGuesser.Model;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text;
using LotteryCore.Tools;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace LotteryCore.Model
{
    public class LotteryHandler
    {
        public Enums.LotteryType LotteryType { get; }
        public GoogleSheetData gsd;
        public  string UserName { get; set; }
        private LotteryRule lotteryRule;
        public  LotteryStatistic LotteryStatistic;

        public  List<SaveNumber> SaveNumbers = new List<SaveNumber>();

         List<LotteryModel> lotteryCollection;

         List<NumberSections> numberSections = new List<NumberSections>();
        private  List<LotteryModel> _lotteryModels;

        public  List<LotteryModel> LotteryModels
        {
            get => _lotteryModels;
            set => _lotteryModels = value;
        }

        public LotteryHandler(Enums.LotteryType lotteryType, string userName,bool isUseGoogleSheet,bool isUseEarlierStatistc, string customLotteryUrl = null)
        {
            LotteryType = lotteryType;
            UserName = userName;
            lotteryRule = new LotteryRule(lotteryType);
            DownloadNumbersFromInternet(lotteryRule.DownloadLink);
            GenerateSections();
            if (isUseGoogleSheet)
            {
                gsd = new GoogleSheetData(UserName);
                LoadNumbersFromSheet(gsd.GetData());
            }
            if(isUseEarlierStatistc) 
                MakeStatisticFromEarlierWeek();

        }

        public void UseGoogleSheet(bool isUseGoogleSheet)
        {
            if (isUseGoogleSheet)
            {
                gsd = new GoogleSheetData(UserName);
                LoadNumbersFromSheet(gsd.GetData());
            }
            else
            {
                gsd = null;
                SaveNumbers = null;
            }
               
           
        }

        public void CalculateNumbers(Enums.TypesOfDrawn tDrawn, Enums.GenerateType generateType, int count)
        {
            //Get the method information using the method info class
            MethodInfo mi = this.GetType().GetMethod(tDrawn.ToString()+"Execute");

            switch (tDrawn)
            {
                
                case Enums.TypesOfDrawn.ByInterval:
                case Enums.TypesOfDrawn.ByOccurrence:
                case Enums.TypesOfDrawn.ByAverageSteps:
                case Enums.TypesOfDrawn.ByAverageRandoms:
                case Enums.TypesOfDrawn.BySums:
                case Enums.TypesOfDrawn.Calculated:
                case Enums.TypesOfDrawn.Test:
                    switch (generateType)
                    {
                        case Enums.GenerateType.EachByEach:
                            RunMethodWithEachTime(mi, count, tDrawn);
                            break;
                        
                        case Enums.GenerateType.GetTheBest:
                            RunMethodWithEachTimeAndGetTheBestNumbers(mi, count, tDrawn);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(generateType), generateType, null);
                    }
                    break;
                case Enums.TypesOfDrawn.All:
                    foreach (Enums.TypesOfDrawn drawn in (Enums.TypesOfDrawn[])Enum.GetValues(typeof(Enums.TypesOfDrawn)))
                    {
                        MethodInfo mis = this.GetType().GetMethod(drawn.ToString() + "Execute");
                        if (mis != null)
                        {
                            switch (generateType)
                            {
                                case Enums.GenerateType.EachByEach:
                                    RunMethodWithEachTime(mis, count, drawn);
                                    break;
                                case Enums.GenerateType.GetTheBest:
                                    RunMethodWithEachTimeAndGetTheBestNumbers(mis, count, drawn);
                                    break;
                                default:
                                    throw new ArgumentOutOfRangeException(nameof(generateType), generateType, null);
                            }
                        }
                    }
                    break;
                case Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw:
                    switch (generateType)
                    {
                        case Enums.GenerateType.EachByEach:
                            break;
                        case Enums.GenerateType.GetTheBest:
                            break;
                        case Enums.GenerateType.Unique:
                            MethodInfo mis = this.GetType().GetMethod("CalcTheFiveMostCommonNumbers");
                            if (mis != null)
                            {
                                RunMethodWithEachTime(mis, count, tDrawn);
                            }
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(generateType), generateType, null);
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(tDrawn), tDrawn, null);
            }


            


            //Invoke the method
            // (null- no parameter for the method call
            // or you can pass the array of parameters...)
            if (mi != null) 
                mi.Invoke(this, null);
        }

        private void RunMethodWithEachTime(MethodInfo invoke, int count, Enums.TypesOfDrawn tDrawn)
        {
            if (LotteryModels == null) LotteryModels = new List<LotteryModel>();
            Console.WriteLine(tDrawn.ToString());
            int index = 0;
            while (true)
            {
                var returnedModel =(LotteryModel) invoke.Invoke(this,null);

                if (returnedModel == null) continue;
                if (LotteryModels.AddValueWithDetailsAndValidation(returnedModel.ValidationTuple(), tDrawn))
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


        public  void DownloadNumbersFromInternet(string link)
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
                    lotteryCollection.Add(new LotteryModel(list, index,lotteryRule));
                    index++;
                }
                
            }
        }

        internal  void LoadNumbersFromSheet(List<string[]> getData)
        {
            SaveNumbers = new List<SaveNumber>();
            if (getData != null && getData.Count > 0)
            {

                foreach (var row in getData)
                {
                    if(row[1].ToString() != "Calculated" && row[3] == UserName && row[4] == lotteryRule.LotteryType.ToString())
                        SaveNumbers.Add(new SaveNumber(row));
                }
            }
            else
            {
                Console.WriteLine("No data found.");
            }
        }

        public void SaveDataToGoogleSheet()
        {
            gsd.SaveNumbersToSheet(LotteryModels);
        }

        public  List<LotteryModel> GetLotteryCollection()
        {
            return lotteryCollection;
        }

        public  List<NumberSections> GetLotteryStatistics()
        {
            return numberSections;
        }

        public  void GenerateSections()
        {
            LotteryStatistic = new LotteryStatistic(lotteryCollection);


            for (int i = lotteryRule.MinNumber; i <= lotteryRule.MaxNumber; i++)
            {
                var actualNumberSection = new NumberSections(i);
                int loti = 1;
                foreach (var t in lotteryCollection)
                {
                    int index = t.Numbers.FindIndex(a => a == i);
                    if (index < lotteryRule.MaxNumber && index >= 0 && loti < lotteryCollection.Count)
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

        public void RunMethodWithEachTimeAndGetTheBestNumbers(MethodInfo action, int count, Enums.TypesOfDrawn tDrawn)
        {
            Dictionary<int,int> numbersDictionary = new Dictionary<int, int>();
            Console.WriteLine($"{tDrawn} {count} Times");
            int index = 0;
            while (index != count)
            {
                LotteryModel returnedModel = (LotteryModel) action.Invoke(this,null);


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

            var sortedDics = numbersDictionary.OrderByDescending(x => x.Value).Take(lotteryRule.PiecesOfDrawNumber);
            LotteryModel resultLotteryModel = new LotteryModel(lotteryRule);
            foreach (KeyValuePair<int, int> keyValuePair in sortedDics)
            {
                resultLotteryModel.Numbers.Add(keyValuePair.Key);
            }

            LotteryModels.AddValueWithDetailsAndValidation(resultLotteryModel.ValidationTuple(),tDrawn);
        }

        public  LotteryModel CalcTheFiveMostCommonNumbers()
        {
            return new LotteryModel(lotteryRule)
            {
                Numbers = LotteryModels.SelectMany(x => x.Numbers).ToList().GroupBy(n => n)
                .Select(n => new
                    {
                        MetricName = n.Key,
                        MetricCount = n.Count()
                    }
                )
                .OrderByDescending(n => n.MetricCount).Take(lotteryRule.PiecesOfDrawNumber).Select(x => x.MetricName).ToList()
            };
        }
        public  LotteryModel ByIntervalExecute()
        {
            
                var acutal = numberSections.Where(x => x.ActualNumber == lotteryCollection.Last().Numbers.First());

                LotteryModel lm = new LotteryModel(lotteryRule);
                for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
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
        public LotteryModel BySumsExecute()
        {
            var getLastSum = lotteryCollection.Last().Sum;

            //find another same sum in list
            LotteryModel getPenultimateSumId;
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
                    getPenultimateSumId = lotteryCollection.FirstOrDefault(x => x.Id == foundLastSum[tek] + 1);
                   
                    break;
                }

                getLastSum++;
            }


            
            LotteryModel lm = new LotteryModel(lotteryRule);
            for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
            {
                lm.AddNumber(GetRandomNumber());
            }
            
            if (lm.Sum >= getPenultimateSumId.Sum - 10 && lm.Sum <= getPenultimateSumId.Sum + 10)
            {
                return lm;
            }

            return null;
        }
        public LotteryModel ByAverageRandomsExecute()
        {
                LotteryModel lm = new LotteryModel(lotteryRule);
                for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
                {
                    var goal = LotteryStatistic.AvarageRandom[i];
                    int id = 0;
                    while (true)
                    {
                        if (id == (int)goal)
                        {
                            lm.AddNumber(GetRandomNumber());
                            break;
                        }
                        id++;
                    }
                }

                return lm;
        }
        public  LotteryModel ByAverageStepsExecute()
        {


                int start2 = GetRandomNumber();
                LotteryModel lm = new LotteryModel(lotteryRule);
                lm.AddNumber(start2);

                foreach (double d in LotteryStatistic.AvarageStepByStep)
                {
                    lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(d, 0));
                }
                
                return lm;

        }

        public int GetRandomNumber()
        {
            Random random = new Random();
            return random.Next(lotteryRule.MinNumber, lotteryRule.MaxNumber+1);
        }

        public  LotteryModel ByOccurrenceExecute()
        {
                LotteryModel lm = new LotteryModel(lotteryRule);



                var getSections = numberSections.Last(x => lotteryCollection.Last().Numbers.First() == x.ActualNumber);
                Random random = new Random();
                SpecifyNumber generateNumber = null;
                int start2 = random.Next(0, getSections.SpecifyNumberList.Count);
                generateNumber = (SpecifyNumber)getSections.SpecifyNumberList[start2].Clone();
                lm.AddNumber(generateNumber.NextNumber);
                for (int k = 0; k < lotteryRule.PiecesOfDrawNumber-1; k++)
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
                            start2 = GetRandomNumber();
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

        public  void MakeStatisticFromEarlierWeek()
        {
            var getLotteryDrawing =  SaveNumbers.Where(x=> x.WeekOfPull ==lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            var lastDrawning = lotteryCollection.Last();
            for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
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

       
        public void UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn tDrawn)
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
                    LotteryModel lm = new LotteryModel(lotteryRule);
                    if (saveNumber.Message != lotteryModel.Message) continue;
                    for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
                    {
                        double calculatedNumber =
                            lotteryModel.Numbers[i] * saveNumber.DifferentInPercentage[i];
                        lm.Numbers.Add((int)calculatedNumber);
                         
                       
                    }
                    LotteryModels.AddValueWithDetailsAndValidation(lm.ValidationTuple(), tDrawn);
                }
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

        /// <summary>
        /// From https://en.wikipedia.org/wiki/ISO_week_date#Weeks_per_year:
        ///
        /// The long years, with 53 weeks in them, can be described by any of the following equivalent definitions:
        ///
        /// - Any year starting on Thursday and any leap year starting on Wednesday.
        /// - Any year ending on Thursday and any leap year ending on Friday.
        /// - Years in which 1 January and 31 December (in common years) or either (in leap years) are Thursdays.
        ///
        /// All other week-numbering years are short years and have 52 weeks.
        /// </summary>
        /// <returns></returns>
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
