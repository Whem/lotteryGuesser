using System;
using System.Collections.Generic;
using System.Text;
using LotteryCore.Tools;

namespace LotteryCore.Model
{
    public class LotteryRule
    {
        public Enums.LotteryType LotteryType { get; }
        public int MinNumber { get; set; }
        public int MaxNumber { get; set; }
        public string DownloadLink { get; set; }
        public int PiecesOfDrawNumber { get; set; }
        public LotteryRule(Enums.LotteryType lotteryType)
        {
            LotteryType = lotteryType;
            switch (lotteryType)
            {
                case Enums.LotteryType.TheFiveNumberDraw:
                    MinNumber = 1;
                    MaxNumber = 90;
                    DownloadLink = "https://bet.szerencsejatek.hu/cmsfiles/otos.html";
                    break;
                case Enums.LotteryType.TheSixNumberDraw:
                    MinNumber = 1;
                    MaxNumber = 45;
                    DownloadLink = "https://bet.szerencsejatek.hu/cmsfiles/hatos.html";
                    
                    break;
                case Enums.LotteryType.Custom:
                    
                    break;
                case Enums.LotteryType.TheSevenNumberDraw:
                    MinNumber = 1;
                    MaxNumber = 35;
                    DownloadLink = "https://bet.szerencsejatek.hu/cmsfiles/skandi.html";
                    break;
                case Enums.LotteryType.EuroJackPot:
                    //TODO: for example euro jackpot
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(lotteryType), lotteryType, null);
            }

            PiecesOfDrawNumber = (int) lotteryType;
        }
    }
}
