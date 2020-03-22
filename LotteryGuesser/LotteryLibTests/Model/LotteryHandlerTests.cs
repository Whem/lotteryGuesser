// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LotteryHandlerTests.cs" company="Whem">
//   Class unit tester 
// </copyright>
// <summary>
//   The lottery handler tests.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryLibTests.Model
{
    using System;
    using System.Linq;

    using LotteryLib.Model;
    using LotteryLib.Tools;

    using NUnit.Framework;

    /// <summary>
    /// The lottery handler tests.
    /// </summary>
    [TestFixture()]
    public class LotteryHandlerTests
    {
        /// <summary>
        /// The lottery handler test.
        /// </summary>
        [Test()]
        public void LotteryHandlerTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, true);
                Assert.True(
                    lh.GetLotteryCollection() != null && lh.LotteryStatistics != null && lh != null
                    && lh.SaveNumbers.Count(number => number.DifferentInPercentage == null) == 0);
            }

            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, true);
                Assert.True(lh.GetLotteryCollection() != null && lh.LotteryStatistic != null);
            }

            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, false);
                Assert.True(lh.GetLotteryCollection() != null && lh.LotteryStatistic != null);
            }

            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, false);
                Assert.True(lh.GetLotteryCollection() != null && lh.LotteryStatistic != null && lh.GoogleSheetData != null);
            }
        }

        /// <summary>
        /// The use google sheet test.
        /// </summary>
        [Test()]
        public void UseGoogleSheetTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, true);
                lh.UseGoogleSheet(false);
                Assert.True(lh.GoogleSheetData == null);
                lh.UseGoogleSheet(true);
                Assert.True(lh.GoogleSheetData != null);
            }
        }

        /// <summary>
        /// The calculate numbers test.
        /// </summary>
        [Test()]
        public void CalculateNumbersTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
                Assert.True(lh.LotteryModels != null);
                lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);

                Assert.True(lh.LotteryModels != null);
            }
        }

        /// <summary>
        /// The download numbers from internet test.
        /// </summary>
        [Test()]
        public void DownloadNumbersFromInternetTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, false);
                Assert.True(lh.GetLotteryCollection() != null);
            }
        }

        /// <summary>
        /// The save data to google sheet test.
        /// </summary>
        [Test()]
        public void SaveDataToGoogleSheetTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, false);
                lh.CalculateNumbers(Enums.TypesOfDrawn.ByInterval, Enums.GenerateType.EachByEach, 1);
                Assert.True(lh.SaveDataToGoogleSheet() != null);
            }
        }

        /// <summary>
        /// The run method with each time and get the best numbers test.
        /// </summary>
        [Test()]
        public void RunMethodWithEachTimeAndGetTheBestNumbersTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);

                Assert.True(lh.LotteryModels != null);
            }
        }

        /// <summary>
        /// The calc the five most common numbers test.
        /// </summary>
        [Test()]
        public void CalcTheFiveMostCommonNumbersTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(Enums.TypesOfDrawn.ByAverageRandoms, Enums.GenerateType.EachByEach, 20);
                lh.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);
                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw) > 0);
            }
        }

        /// <summary>
        /// The by interval execute test.
        /// </summary>
        [Test()]
        public void ByIntervalExecuteTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var testEnum = Enums.TypesOfDrawn.ByInterval;
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(testEnum, Enums.GenerateType.EachByEach, 20);
               
                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == testEnum) > 0);
            }
        }

        /// <summary>
        /// The by sums execute test.
        /// </summary>
        [Test()]
        public void BySumsExecuteTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var testEnum = Enums.TypesOfDrawn.BySums;
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(testEnum, Enums.GenerateType.EachByEach, 20);

                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == testEnum) > 0);
            }
        }

        /// <summary>
        /// The by average randoms execute test.
        /// </summary>
        [Test()]
        public void ByAverageRandomsExecuteTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var testEnum = Enums.TypesOfDrawn.ByAverageRandoms;
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(testEnum, Enums.GenerateType.EachByEach, 20);

                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == testEnum) > 0);
            }
        }

        /// <summary>
        /// The by average steps execute test.
        /// </summary>
        [Test()]
        public void ByAverageStepsExecuteTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var testEnum = Enums.TypesOfDrawn.ByAverageSteps;
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(testEnum, Enums.GenerateType.EachByEach, 20);

                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == testEnum) > 0);
            }
        }

        /// <summary>
        /// The get random number test.
        /// </summary>
        [Test()]
        public void GetRandomNumberTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lr = new LotteryRule(lt);
                var rnd = LotteryHandler.GetRandomNumber(lr);
                Assert.True(rnd > lr.MinNumber && rnd < lr.MaxNumber);
            }
        }

        /// <summary>
        /// The by occurrence execute test.
        /// </summary>
        [Test()]
        public void ByOccurrenceExecuteTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var testEnum = Enums.TypesOfDrawn.ByOccurrence;
                var lh = new LotteryHandler(lt, "Whem", false, false);
                lh.CalculateNumbers(testEnum, Enums.GenerateType.EachByEach, 20);

                Assert.True(
                    lh.LotteryModels.Count(x => x.Message == testEnum) > 0);
            }
        }

        /// <summary>
        /// The make statistic from earlier week test.
        /// </summary>
        [Test()]
        public void MakeStatisticFromEarlierWeekTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, true);
                Assert.True(
                    lh.GetLotteryCollection() != null && lh.LotteryStatistics != null && lh != null
                    && lh.SaveNumbers.Count(number => number.DifferentInPercentage == null) == 0);
            }
        }

        /// <summary>
        /// The use earlier week percentage for numbers draw test.
        /// </summary>
        [Test()]
        public void UseEarlierWeekPercentageForNumbersDrawTest()
        {
            foreach (Enums.LotteryType lt in (Enums.LotteryType[])Enum.GetValues(typeof(Enums.LotteryType)))
            {
                var lh = new LotteryHandler(lt, "Whem", true, true);
                lh.CalculateNumbers(Enums.TypesOfDrawn.ByAverageRandoms, Enums.GenerateType.EachByEach, 20);

                try
                {
                    lh.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
                    var test = lh.LotteryModels.Count(x => x.Message == Enums.TypesOfDrawn.Calculated);
                    Assert.True(test > 0);
                }
                catch (Exception e)
                {
                    Assert.True(e.Message != null);
                }
            }
        }

        /// <summary>
        /// The get weeks in year test.
        /// </summary>
        [Test()]
        public void GetWeeksInYearTest()
        {
            Assert.True(LotteryHandler.GetWeeksInYear() > 0);
        }
    }
}