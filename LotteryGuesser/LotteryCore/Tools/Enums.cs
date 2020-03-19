using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryCore.Tools
{
    public class Enums
    {
        public enum TypesOfDrawn
        {
            ByInterval,
            ByOccurrence,
            ByAverageSteps,
            ByAverageRandoms,
            BySums,
            Calculated,
            ByDistributionBasedCurrentDraw,
            Test,
            All
        }

        public enum GenerateType
        {
            EachByEach,
            GetTheBest,
            Unique
        }

        public enum LotteryType
        {
            TheFiveNumberDraw = 5,
            TheSixNumberDraw = 6,
            TheSevenNumberDraw = 7,
            EuroJackPot,
            Custom,
        }
    }
}
