using System;
using System.Collections.Generic;
using System.Linq;

namespace LotteryCore.Model
{
    public class LotteryStatistic
    {
        public List<double> AvarageStepByStep { get; set; }
        

        public List<double> AvarageRandom { get; set; }

        public List<LotteryModel> SameDraw { get; set; }

        public List<IntervallNumber> IntervallNumbers { get; set; }

        public LotteryStatistic(List<LotteryModel> lotteryModels)
        {
            AvarageRandom = new List<double>();
            SameDraw = new List<LotteryModel>();
            IntervallNumbers = new List<IntervallNumber>();
            AvarageStepByStep = new List<double>();
            
            for (int i = 0; i < lotteryModels[0].LotteryRule.PiecesOfDrawNumber; i++)
            {
                if(lotteryModels[0].LotteryRule.PiecesOfDrawNumber-1 >i)
                    AvarageStepByStep.Add(lotteryModels.Select(x => x.Avarages[i]).Average());
                AvarageRandom.Add(lotteryModels.Select(x => x.RandomToGetNumber[i]).Average());
            }

            foreach (LotteryModel lotteryModel in lotteryModels)
            {               
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
