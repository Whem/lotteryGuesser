// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Enums.cs" company="Whem">
//   Enum class
// </copyright>
// <summary>
//   Defines the Enums type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryMaui.Model.Lottery.Tools
{
    /// <summary>
    /// The enums.
    /// </summary>
    public partial class Enums
    {
        /// <summary>
        /// The generate type.
        /// </summary>
        public enum GenerateType
        {
            /// <summary>
            /// The each by each.
            /// </summary>
            EachByEach,

            /// <summary>
            /// The get the best.
            /// </summary>
            GetTheBest,

            /// <summary>
            /// The unique.
            /// </summary>
            Unique,

            MostCommonSeries,

            Calculate,
        }

        /// <summary>
        /// The lottery type.
        /// </summary>
        public enum LotteryType
        {
            /// <summary>
            /// The the five number draw.
            /// </summary>
            TheFiveNumberDraw = 5,

            /// <summary>
            /// The the six number draw.
            /// </summary>
            TheSixNumberDraw = 6,

            /// <summary>
            /// The the seven number draw.
            /// </summary>
            TheSevenNumberDraw = 7,

            // EuroJackPot,
            // Custom,
        }
    }
}
