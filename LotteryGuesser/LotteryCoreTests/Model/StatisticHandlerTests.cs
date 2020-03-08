using Microsoft.VisualStudio.TestTools.UnitTesting;
using LotteryCore.Model;
using System;
using System.Collections.Generic;
using System.Text;
using LotteryCore.Tools;

namespace LotteryCore.Model.Tests
{
    [TestClass()]
    public class StatisticHandlerTests
    {
        public List<LotteryModel> TestLotteryModels;

        public StatisticHandlerTests()
        {
            TestLotteryModels = new List<LotteryModel>
            {
                new LotteryModel()
                {
                    Numbers = new List<int>()
                    {
                        2,
                        3,
                        4,
                        5,
                        6
                    }
                },
                new LotteryModel()
                {
                    Numbers = new List<int>()
                    {
                        1,
                        2,
                        3,
                        4,
                        5
                    }
                }
            };
        }

        [TestMethod()]
        public void DownloadNumbersFromInternetTest()
        {
            try
            {
                var lh = new LotteryHandler();
                lh.DownloadNumbersFromInternet("https://bet.szerencsejatek.hu/cmsfiles/otos.html");
                Assert.IsFalse(lh.GetLotteryCollection() == null || lh.GetLotteryCollection().Count == 0);

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        
        [TestMethod()]
        public void GetLotteryCollectionTest()
        {
            try
            {
                var lh = new LotteryHandler();
                lh.GenerateSections();
                Assert.IsFalse(lh.LotteryStatistic == null);

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        

        [TestMethod()]
        public void LoadNumbersFromJsonTest()
        {
            try
            {
                var gsd = new GoogleSheetData();

                Assert.IsFalse(gsd.GetData() == null);
            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
           
           
        }

        [TestMethod()]
        public void GenerateSectionsTest()
        {
            try
            {
                var lh = new LotteryHandler();
                lh.GenerateSections();
                Assert.IsFalse(lh.GetLotteryStatistics() == null);
            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
           
        }

        [TestMethod()]
        public void RunMethodWithEachTimeTest()
        {
            //try
            //{
            //    LotteryHandler.RunMethodWithEachTime(LotteryHandler.GenerateAverageStepLines,1, Enums.TypesOfDrawn.Test);
            //    Assert.IsFalse(LotteryHandler.LotteryModels.Count ==0 || LotteryHandler.LotteryModels == null);
            //}
            //catch (Exception e)
            //{
            //    Assert.Fail(e.Message);
            //}
        }

        [TestMethod()]
        public void SaveCurrentNumbersToFileWithJsonTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GenerateFromInterValTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GenerateNumbersFromSumTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GenereateRandomTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GenerateAvarageStepLinesTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GenerateLotteryTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void MakeStatisticFromEarlierWeekTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void UseEarlierWeekPercentageForNumbersDrawTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void GetWeeksInYearTest()
        {
            try
            {

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }
    }
}