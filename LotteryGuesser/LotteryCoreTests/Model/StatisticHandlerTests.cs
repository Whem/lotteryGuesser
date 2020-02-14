using Microsoft.VisualStudio.TestTools.UnitTesting;
using LotteryCore.Model;
using System;
using System.Collections.Generic;
using System.Text;

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
                StatisticHandler.DownloadNumbersFromInternet("https://bet.szerencsejatek.hu/cmsfiles/otos.html");
                Assert.IsFalse(StatisticHandler.GetLotteryCollection() == null || StatisticHandler.GetLotteryCollection().Count == 0);

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
                StatisticHandler.GenerateSections();
                Assert.IsFalse(StatisticHandler.LotteryStatistic == null);

            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
        }

        [TestMethod()]
        public void AddNumbersToSaveFileTest()
        {
            try
            {
                StatisticHandler.AddNumbersToSaveFile(new SaveNumber(new int[]{1,2,3,4,5},"test"));
                Assert.IsFalse(StatisticHandler.SaveNumbers.Count <1);
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
                StatisticHandler.LoadNumbersFromJson("test.json");
                Assert.IsFalse(StatisticHandler.SaveNumbers == null);
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
                StatisticHandler.GenerateSections();
                Assert.IsFalse(StatisticHandler.GetLotteryStatistics() == null);
            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
           
        }

        [TestMethod()]
        public void RunMethodWithEachTimeTest()
        {
            try
            {
                StatisticHandler.RunMethodWithEachTime(StatisticHandler.GenerateAvarageStepLines,1,"Test");
                Assert.IsFalse(StatisticHandler.LotteryModels.Count ==0 || StatisticHandler.LotteryModels == null);
            }
            catch (Exception e)
            {
                Assert.Fail(e.Message);
            }
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