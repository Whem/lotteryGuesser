// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LotteryRule.cs" company="Whem">
//   LotteryRule  class make rules to lottery drawning, and how it is working it in real life
// </copyright>
// <summary>
//   The lottery rule.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

using LotteryMaui.Model.Lottery.Tools;

namespace LotteryMaui.Model.Lottery.Model
{
    /// <summary>
    /// The lottery rule.
    /// </summary>
    public class LotteryRule
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LotteryRule"/> class.
        /// </summary>
        /// <param name="lotteryType">
        /// The lottery type.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// </exception>
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
                case Enums.LotteryType.TheSevenNumberDraw:
                    MinNumber = 1;
                    MaxNumber = 35;
                    DownloadLink = "https://bet.szerencsejatek.hu/cmsfiles/skandi.html";
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(lotteryType), lotteryType, null);
            }

            PiecesOfDrawNumber = (int)lotteryType;
        }

        /// <summary>
        /// Gets the lottery type.
        /// </summary>
        public Enums.LotteryType LotteryType { get; }

        /// <summary>
        /// Gets or sets the min number.
        /// </summary>
        public int MinNumber { get; set; }

        /// <summary>
        /// Gets or sets the max number.
        /// </summary>
        public int MaxNumber { get; set; }

        /// <summary>
        /// Gets or sets the download link.
        /// </summary>
        public string DownloadLink { get; set; }

        /// <summary>
        /// Gets or sets the pieces of draw number.
        /// </summary>
        public int PiecesOfDrawNumber { get; set; }
    }
}
