using System;
using System.Collections.Generic;
using System.Linq;

namespace LotteryCore.Model
{
    public class LotteryStatistic
    {
        public double Avarage1to2 { get; set; }
        public double Avarage2to3 { get; set; }
        public double Avarage3to4 { get; set; }
        public double Avarage4to5 { get; set; }

        public List<double> AvarageRandom { get; set; }

        public List<LotteryModel> SameDraw { get; set; }

        public List<IntervallNumber> IntervallNumbers { get; set; }

        public LotteryStatistic(List<LotteryModel> lotteryModels)
        {
            AvarageRandom = new List<double>();
            SameDraw = new List<LotteryModel>();
            IntervallNumbers = new List<IntervallNumber>();
            Avarage1to2 = lotteryModels.Select(x => x.Avarages[0]).Average();
            Avarage2to3 = lotteryModels.Select(x => x.Avarages[1]).Average();
            Avarage3to4 = lotteryModels.Select(x => x.Avarages[2]).Average();
            Avarage4to5 = lotteryModels.Select(x => x.Avarages[3]).Average();

            AvarageRandom.Add(lotteryModels.Select(x=> x.RandomToGetNumber[0]).Average());
            AvarageRandom.Add(lotteryModels.Select(x => x.RandomToGetNumber[1]).Average());
            AvarageRandom.Add(lotteryModels.Select(x => x.RandomToGetNumber[2]).Average());
            AvarageRandom.Add(lotteryModels.Select(x => x.RandomToGetNumber[3]).Average());
            AvarageRandom.Add(lotteryModels.Select(x => x.RandomToGetNumber[4]).Average());

            foreach (LotteryModel lotteryModel in lotteryModels)
            {
                var strings = lotteryModel.ToString();
                SameDraw.AddRange(lotteryModels.Where( x=> x.Sum == lotteryModel.Sum && x.Id != lotteryModel.Id ).ToList());
            }

            IntervallNumbers.Add(new IntervallNumber(1,10));
            IntervallNumbers.Add(new IntervallNumber(11, 20));
            IntervallNumbers.Add(new IntervallNumber(21, 30));
            IntervallNumbers.Add(new IntervallNumber(31, 40));
            IntervallNumbers.Add(new IntervallNumber(41, 50));
            IntervallNumbers.Add(new IntervallNumber(51, 60));
            IntervallNumbers.Add(new IntervallNumber(61, 70));
            IntervallNumbers.Add(new IntervallNumber(71, 80));
            IntervallNumbers.Add(new IntervallNumber(81, 90));
        }
    }
}
