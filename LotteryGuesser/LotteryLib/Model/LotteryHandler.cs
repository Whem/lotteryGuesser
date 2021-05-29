// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LotteryHandler.cs" company="Whem">
//   THis is the main class of everything
// </copyright>
// <summary>
//   The lottery handler.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryLib.Model
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.Linq;
    using System.Net;
    using System.Reflection;

    using Encog.Engine.Network.Activation;
    using Encog.ML.Data.Basic;
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Layers;
    using Encog.Neural.Networks.Training.Propagation.Resilient;

    using Google.Apis.Sheets.v4.Data;

    using LotteryLib.Tools;

    /// <summary>
    /// The lottery handler.
    /// </summary>
    public class LotteryHandler
    {
        /// <summary>
        /// The number sections.
        /// </summary>
        public readonly List<NumberSections> NumberSections = new List<NumberSections>();

        /// <summary>
        /// The lottery rule.
        /// </summary>
        private readonly LotteryRule lotteryRule;

        /// <summary>
        /// The lottery collection.
        /// </summary>
        public List<LotteryModel> lotteryCollection;

        /// <summary>
        /// Initializes a new instance of the <see cref="LotteryHandler"/> class.
        /// </summary>
        /// <param name="lotteryType">
        /// The lottery type.
        /// </param>
        /// <param name="userName">
        /// The user name.
        /// </param>
        /// <param name="isUseGoogleSheet">
        /// The is use google sheet.
        /// </param>
        /// <param name="isUseEarlierStatistic">
        /// The is use earlier statistic.
        /// </param>
        public LotteryHandler(Enums.LotteryType lotteryType, string userName, bool isUseGoogleSheet, bool isUseEarlierStatistic)
        {
            LotteryType = lotteryType;
            UserName = userName;
            lotteryRule = new LotteryRule(lotteryType);
            DownloadNumbersFromInternet(lotteryRule.DownloadLink);
            GenerateSections();
            if (isUseGoogleSheet)
            {
                GoogleSheetData = new GoogleSheetData(UserName);
                LoadNumbersFromSheet(GoogleSheetData.GetData());
                if (isUseEarlierStatistic)
                    MakeStatisticFromEarlierWeek();
            }
        }

        /// <summary>
        /// The lottery model event.
        /// </summary>
        public static event EventHandler<LotteryModel> LotteryModelEvent;

        /// <summary>
        /// Gets the lottery type.
        /// </summary>
        public Enums.LotteryType LotteryType { get; }

        /// <summary>
        /// Gets or sets the user name.
        /// </summary>
        public string UserName { get; set; }

        /// <summary>
        /// Gets or sets the lottery models.
        /// </summary>
        public List<LotteryModel> LotteryModels { get; set; }

        /// <summary>
        /// Gets or sets the save numbers.
        /// </summary>
        public List<SaveNumber> SaveNumbers { get; set; }

        /// <summary>
        /// Gets or sets the google sheet data.
        /// </summary>
        public GoogleSheetData GoogleSheetData { get; set; }

        /// <summary>
        /// Gets or sets the lottery statistic.
        /// </summary>
        public LotteryStatistic LotteryStatistic { get; set; }

        /// <summary>
        /// The lottery statistics.
        /// </summary>
        public List<NumberSections> LotteryStatistics => NumberSections;

        /// <summary>
        /// From https://en.wikipedia.org/wiki/ISO_week_date#Weeks_per_year:
        /// The long years, with 53 weeks in them, can be described by any of the following equivalent definitions:
        /// -Any year starting on Thursday and any leap year starting on Wednesday.
        /// -Any year ending on Thursday and any leap year ending on Friday.
        /// -Years in which 1 January and 31 December (in common years) or either (in leap years) are Thursdays.
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

        /// <summary>
        /// The get random number.
        /// </summary>
        /// <param name="lotteryRule">
        /// The lottery Rule.
        /// </param>
        /// <returns>
        /// The <see cref="int"/>.
        /// </returns>
        public static int GetRandomNumber(LotteryRule lotteryRule)
        {
            Random random = new Random();
            return random.Next(lotteryRule.MinNumber, lotteryRule.MaxNumber + 1);
        }

        /// <summary>
        /// The use google sheet.
        /// </summary>
        /// <param name="isUseGoogleSheet">
        /// The is use google sheet.
        /// </param>
        public void UseGoogleSheet(bool isUseGoogleSheet)
        {
            if (isUseGoogleSheet)
            {
                GoogleSheetData = new GoogleSheetData(UserName);
                LoadNumbersFromSheet(this.GoogleSheetData.GetData());
            }
            else
            {
                this.GoogleSheetData = null;
                this.SaveNumbers = null;
            }             
        }

        /// <summary>
        /// The download numbers from internet.
        /// </summary>
        /// <param name="link">
        /// The link.
        /// </param>
        public void DownloadNumbersFromInternet(string link)
        {
            this.lotteryCollection = new List<LotteryModel>();
            using (WebClient client = new WebClient())
            {
                // WebClient class inherits IDisposable
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
                    // ReSharper disable once ArrangeThisQualifier
                    this.lotteryCollection.Add(new LotteryModel(list, index, lotteryRule));
                    index++;
                }
            }
        }

        /// <summary>
        /// The calculate numbers.
        /// </summary>
        /// <param name="typesOfDrawn">
        /// The t drawn.
        /// </param>
        /// <param name="generateType">
        /// The generate type.
        /// </param>
        /// <param name="count">
        /// The count.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// </exception>
        public void CalculateNumbers(Enums.TypesOfDrawn typesOfDrawn, Enums.GenerateType generateType, int count)
        {
            // Get the method information using the method info class
            MethodInfo mi = this.GetType().GetMethod(typesOfDrawn + "Execute");

            switch (typesOfDrawn)
            {
                case Enums.TypesOfDrawn.ByInterval:
                case Enums.TypesOfDrawn.ByOccurrence:
                case Enums.TypesOfDrawn.ByAverageSteps:
                case Enums.TypesOfDrawn.ByAverageRandoms:
                case Enums.TypesOfDrawn.BySums:
                case Enums.TypesOfDrawn.Calculated:
                case Enums.TypesOfDrawn.ByMachineLearning:
                    switch (generateType)
                    {
                        case Enums.GenerateType.EachByEach:
                            this.RunMethodWithEachTime(mi, count, typesOfDrawn);
                            break;
                        
                        case Enums.GenerateType.GetTheBest:
                            this.RunMethodWithEachTimeAndGetTheBestNumbers(mi, count, typesOfDrawn);
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
                                RunMethodWithEachTime(mis, count, typesOfDrawn);
                            }

                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(generateType), generateType, null);
                    }

                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(typesOfDrawn), typesOfDrawn, null);
            }

            // Invoke the method
            // (null- no parameter for the method call
            // or you can pass the array of parameters...)
            if (mi != null) mi.Invoke(this, null);
        }

        /// <summary>
        /// The generate sections.
        /// </summary>
        public void GenerateSections()
        {
            this.LotteryStatistic = new LotteryStatistic(this.lotteryCollection);

            for (int i = lotteryRule.MinNumber; i <= lotteryRule.MaxNumber; i++)
            {
                var actualNumberSection = new NumberSections(i);
                int lot = 1;
                foreach (var t in this.lotteryCollection)
                {
                    int index = t.Numbers.FindIndex(a => a == i);
                    if (index < lotteryRule.MaxNumber && index >= 0 && lot < this.lotteryCollection.Count)
                    {
                        actualNumberSection.FindTheNextNumber(this.lotteryCollection[lot].Numbers[index]);

                        var getInterval = this.LotteryStatistic.IntervallNumbers.First(x => x.StartInterVal <= this.lotteryCollection[lot].Numbers[index] && x.StopInterval >= this.lotteryCollection[lot].Numbers[index]);
                        getInterval.ActualNumberList.Add(t.Numbers[index]);
                        getInterval.AfterNumberList.Add(this.lotteryCollection[lot].Numbers[index]);
                    }

                    lot++;
                }

                this.NumberSections.Add(actualNumberSection);
            }
        }

        /// <summary>
        /// The run method with each time and get the best numbers.
        /// </summary>
        /// <param name="action">
        /// The action.
        /// </param>
        /// <param name="count">
        /// The count.
        /// </param>
        /// <param name="typesOfDrawn">
        /// The t drawn.
        /// </param>
        public void RunMethodWithEachTimeAndGetTheBestNumbers(MethodInfo action, int count, Enums.TypesOfDrawn typesOfDrawn)
        {
            if (LotteryModels == null) LotteryModels = new List<LotteryModel>();
            Dictionary<int, int> numbersDictionary = new Dictionary<int, int>();
            Console.WriteLine($"{typesOfDrawn} {count} Times");
            int index = 0;
            while (index != count)
            {
                LotteryModel returnedModel = (LotteryModel)action.Invoke(this, null);
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

            var sortedDic = numbersDictionary.OrderByDescending(x => x.Value).Take(lotteryRule.PiecesOfDrawNumber);
            LotteryModel resultLotteryModel = new LotteryModel(lotteryRule);
            foreach (KeyValuePair<int, int> keyValuePair in sortedDic)
            {
                resultLotteryModel.Numbers.Add(keyValuePair.Key);
            }

            if (LotteryModels.AddValueWithDetailsAndValidation(resultLotteryModel.ValidationTuple(), typesOfDrawn))
            {
                OnLotteryModelEvent(resultLotteryModel);
            }
        }

        /// <summary>
        /// The calc the five most common numbers.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel CalcTheFiveMostCommonNumbers()
        {
            return new LotteryModel(lotteryRule)
            {
                Numbers = LotteryModels.SelectMany(x => x.Numbers).ToList().GroupBy(n => n)
                .Select(n => new
                    {
                        MetricName = n.Key,
                        MetricCount = n.Count()
                    }).OrderByDescending(n => n.MetricCount).Take(lotteryRule.PiecesOfDrawNumber)
                .Select(x => x.MetricName).ToList()
            };
        }

        public LotteryModel ByMachineLearningExecute()
        {
            try
            {
                if (this.lotteryRule.LotteryType != Enums.LotteryType.TheFiveNumberDraw) return null; //TODO: Make this for 6 and custom lottery draw
                LotteryModel lm = new LotteryModel(lotteryRule);
                var dbl = new List<LotteryResult>();
                foreach (LotteryModel lotteryModel in this.lotteryCollection)
                {
                    var res = new LotteryResult(
                        lotteryModel.Numbers[0],
                        lotteryModel.Numbers[1],
                        lotteryModel.Numbers[2],
                        lotteryModel.Numbers[3],
                        lotteryModel.Numbers[4]
                    );

                    dbl.Add(res);
                }

                dbl.Reverse();
                var deep = 20;
                var network = new BasicNetwork();
                network.AddLayer(
                new BasicLayer(null, true, 5 * deep));
                network.AddLayer(
                new BasicLayer(
                new ActivationSigmoid(), true, 4 * 5 * deep));
                network.AddLayer(
                new BasicLayer(
                new ActivationSigmoid(), true, 4 * 5 * deep));
                network.AddLayer(
                new BasicLayer(
                new ActivationLinear(), true, 5));
                network.Structure.FinalizeStructure();
                var learningInput = new double[deep][];
                for (int i = 0; i < deep; ++i)
                {
                    learningInput[i] = new double[deep * 5];
                    for (int j = 0, k = 0; j < deep; ++j)
                    {
                        var idx = 2 * deep - i - j;
                        LotteryResult data = dbl[idx];
                        learningInput[i][k++] = (double)data.V1;
                        learningInput[i][k++] = (double)data.V2;
                        learningInput[i][k++] = (double)data.V3;
                        learningInput[i][k++] = (double)data.V4;
                        learningInput[i][k++] = (double)data.V5;
                    }
                }
                var learningOutput = new double[deep][];
                for (int i = 0; i < deep; ++i)
                {
                    var idx = deep - 1 - i;
                    var data = dbl[idx];
                    learningOutput[i] = new double[5]
                    {
                        (double)data.V1,
                        (double)data.V2,
                        (double)data.V3,
                        (double)data.V4,
                        (double)data.V5
                    };
                }
                var trainingSet = new BasicMLDataSet(
                learningInput,
                learningOutput);

                var train = new ResilientPropagation(
                network, trainingSet);
                train.NumThreads = Environment.ProcessorCount;
                START:
                network.Reset();
                RETRY:
                var step = 0;
                do
                {
                    train.Iteration();
                    Console.WriteLine("Train Error: {0}", train.Error);
                    ++step;
                }
                while (train.Error > 0.001 && step < 20);
                var passedCount = 0;
                for (var i = 0; i < deep; ++i)
                {
                    var should =
                    new LotteryResult(learningOutput[i]);
                    var inputn = new BasicMLData(5 * deep);
                    Array.Copy(
                    learningInput[i],
                    inputn.Data,
                    inputn.Data.Length);
                    var comput =
                    new LotteryResult(
                    ((BasicMLData)network.
                    Compute(inputn)).Data);
                    var passed = should.ToString() == comput.ToString();
                    if (passed)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        ++passedCount;
                    }
                }

                var input = new BasicMLData(5 * deep);
                for (int i = 0, k = 0; i < deep; ++i)
                {
                    var idx = deep - 1 - i;
                    var data = dbl[idx];
                    input.Data[k++] = (double)data.V1;
                    input.Data[k++] = (double)data.V2;
                    input.Data[k++] = (double)data.V3;
                    input.Data[k++] = (double)data.V4;
                    input.Data[k++] = (double)data.V5;
                }
                var perfect = dbl[0];
                LotteryResult predict = new LotteryResult(
                ((BasicMLData)network.Compute(input)).Data);
                Console.WriteLine("Predict: {0}", predict);
                
                if (predict.IsOut())
                    goto START;

                var t = passedCount < (deep * (double)9 / (double)10);
                var isvalid = predict.IsValid();

                if (t ||
                  !isvalid)
                    goto RETRY;
                
                lm.AddNumber(predict.V1);
                lm.AddNumber(predict.V2);
                lm.AddNumber(predict.V3);
                lm.AddNumber(predict.V4);
                lm.AddNumber(predict.V5);

                return lm;
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.ToString());
                return null;
            }
        }



        /// <summary>
        /// The by interval execute.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel ByIntervalExecute()
        {
                var actual = this.NumberSections.Where(x => x.ActualNumber == this.lotteryCollection.Last().Numbers.First());

                LotteryModel lm = new LotteryModel(lotteryRule);
                for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
                {
                    int max = 0;
                    List<IOrderedEnumerable<SpecifyNumber>> spec = actual.Select(x => x.SpecifyNumberList.OrderByDescending(xy => xy.Pieces)).ToList();
                    List<int> maxCount = spec[0].Select(x => x.Pieces).OrderByDescending(x => x).ToList();
                    SpecifyNumber getSec = spec[0].First(x => x.Pieces == maxCount[max]);

                    while (true)
                    {
                        if (lm.Numbers.Contains(getSec.NextNumber))
                        {
                            max++;
                            getSec = spec[0].First(x => x.Pieces == maxCount[max]);
                        }
                        else
                        {
                            if (lm.Numbers.Count == 0)
                                break;
                            if (getSec.NextNumber > lm.Numbers.Max())
                                break;
                            max++;
                            getSec = spec[0].First(x => x.Pieces == maxCount[max]);
                        }
                    }

                    var getAf = getSec.NextNumber;
                    lm.Numbers.Add(getAf);
                    actual = this.NumberSections.Where(x => x.ActualNumber == getAf);
                }

                return lm;
        }

        /// <summary>
        /// The by sums execute.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel BySumsExecute()
        {
            var getLastSum = this.lotteryCollection.Last().Sum;

            // find another same sum in list
            LotteryModel getPenultimateSumId;
            while (true)
            {
                Random rnd = new Random();
                var foundLastSum = lotteryCollection.Select((v, i) => new { Index = i, Value = v }).Where(x => x.Value.Sum == getLastSum)
                    .Select(x => x.Value.Id)
                    .ToList();
                if (foundLastSum.Count > 2)
                {
                    var tek = rnd.Next(1, foundLastSum.Count - 1);
                    getPenultimateSumId = this.lotteryCollection.FirstOrDefault(x => x.Id == foundLastSum[tek] + 1);
                   
                    break;
                }

                getLastSum++;
            }

            LotteryModel lm = new LotteryModel(lotteryRule);
            for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
            {
                lm.AddNumber(GetRandomNumber(lotteryRule));
            }
            
            if (getPenultimateSumId != null && (lm.Sum >= getPenultimateSumId.Sum - 10 && lm.Sum <= getPenultimateSumId.Sum + 10))
            {
                return lm;
            }

            return null;
        }

        /// <summary>
        /// The by average randoms execute.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel ByAverageRandomsExecute()
        {
                LotteryModel lm = new LotteryModel(lotteryRule);
                for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
                {
                    var goal = this.LotteryStatistic.AvarageRandom[i];
                    int id = 0;
                    while (true)
                    {
                        if (id == (int)goal)
                        {
                            lm.AddNumber(GetRandomNumber(lotteryRule));
                            break;
                        }

                        id++;
                    }
                }

                return lm;
        }

        /// <summary>
        /// The by average steps execute.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel ByAverageStepsExecute()
        {
                LotteryModel lm = new LotteryModel(lotteryRule);
                int start2 = GetRandomNumber(lotteryRule);
                lm.AddNumber(start2);

                foreach (double d in this.LotteryStatistic.AvarageStepByStep)
                {
                    lm.AddNumber(lm.Numbers.Last() + (int)Math.Round(d, 0));
                }
                
                return lm;
        }

        /// <summary>
        /// The by occurrence execute.
        /// </summary>
        /// <returns>
        /// The <see cref="LotteryModel"/>.
        /// </returns>
        public LotteryModel ByOccurrenceExecute()
        {
                LotteryModel lm = new LotteryModel(lotteryRule);

                var getSections = this.NumberSections.Last(x => this.lotteryCollection.Last().Numbers.First() == x.ActualNumber);
                Random random = new Random();
                int start2 = random.Next(0, getSections.SpecifyNumberList.Count);
                var generateNumber = (SpecifyNumber)getSections.SpecifyNumberList[start2].Clone();
                lm.AddNumber(generateNumber.NextNumber);
                for (int k = 0; k < lotteryRule.PiecesOfDrawNumber - 1; k++)
                {
                    getSections = this.NumberSections.First(x => lm.Numbers.Last() == x.ActualNumber);

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
                            start2 = GetRandomNumber(lotteryRule);
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

        /// <summary>
        /// The make statistic from earlier week.
        /// </summary>
        public void MakeStatisticFromEarlierWeek()
        {
            List<SaveNumber> getLotteryDrawing = this.SaveNumbers
                .Where(x => x.WeekOfPull == this.lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            LotteryModel lastDrawing = this.lotteryCollection.Last();
            for (int i = 0; i < lotteryRule.PiecesOfDrawNumber; i++)
            {
                foreach (var t1 in getLotteryDrawing)
                {
                    var e = t1.Numbers[i];
                    var k = lastDrawing.Numbers[i];
                    var t = (double)e / k;
                    t1.DifferentInPercentage.Add(t);
                }
            }
        }

        /// <summary>
        /// The use earlier week percentage for numbers draw.
        /// </summary>
        /// <param name="typesOfDrawn">
        /// The t drawn.
        /// </param>
        public void UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn typesOfDrawn)
        {
            List<SaveNumber> getLotteryDrawing = this.SaveNumbers.Where(x => x.WeekOfPull == this.lotteryCollection.Last().WeekOfLotteryDrawing).ToList();
            //if (getLotteryDrawing.Count == 0)
            //{
            //    throw new InvalidOperationException("You haven't earlier week result");
            //}

            if (LotteryModels == null) throw new InvalidOperationException("You didn't generate numbers from which I can calculate");
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

                    LotteryModels.AddValueWithDetailsAndValidation(lm.ValidationTuple(), typesOfDrawn);
                }
            }
        }

        /// <summary>
        /// The save data to google sheet.
        /// </summary>
        /// <returns>
        /// The <see cref="AppendValuesResponse"/>.
        /// </returns>
        public AppendValuesResponse SaveDataToGoogleSheet()
        {
            return GoogleSheetData.SaveNumbersToSheet(LotteryModels);
        }

        /// <summary>
        /// The get lottery collection.
        /// </summary>
        /// <returns>
        /// The <see cref="lotteryCollection"/>.
        /// </returns>
        public List<LotteryModel> GetLotteryCollection()
        {
            return lotteryCollection;
        }

        /// <summary>
        /// The load numbers from sheet.
        /// </summary>
        /// <param name="getData">
        /// The get data.
        /// </param>
        internal void LoadNumbersFromSheet(List<string[]> getData)
        {
            this.SaveNumbers = new List<SaveNumber>();
            if (getData != null && getData.Count > 0)
            {
                foreach (var row in getData)
                {
                    if (row[1] != "Calculated" && row[3] == UserName && row[4] == lotteryRule.LotteryType.ToString())
                        this.SaveNumbers.Add(new SaveNumber(row));
                }
            }
            else
            {
                Console.WriteLine("No data found.");
            }
        }

        /// <summary>
        /// The run method with each time.
        /// </summary>
        /// <param name="invoke">
        /// The invoke.
        /// </param>
        /// <param name="count">
        /// The count.
        /// </param>
        /// <param name="typesOfDrawn">
        /// The t drawn.
        /// </param>
        private void RunMethodWithEachTime(MethodInfo invoke, int count, Enums.TypesOfDrawn typesOfDrawn)
        {
            if (LotteryModels == null) LotteryModels = new List<LotteryModel>();
            Console.WriteLine(typesOfDrawn.ToString());
            int index = 0;
            while (true)
            {
                var returnedModel = (LotteryModel)invoke.Invoke(this, null);

                if (returnedModel == null) continue;
                if (LotteryModels.AddValueWithDetailsAndValidation(returnedModel.ValidationTuple(), typesOfDrawn))
                {
                    index++;
                    OnLotteryModelEvent(returnedModel);
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

        /// <summary>
        /// The on lottery model event.
        /// </summary>
        /// <param name="e">
        /// The e.
        /// </param>
        private static void OnLotteryModelEvent(LotteryModel e)
        {
            LotteryModelEvent?.Invoke(null, e);
        }
    }
}
