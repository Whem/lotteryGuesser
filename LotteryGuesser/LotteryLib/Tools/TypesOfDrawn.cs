// --------------------------------------------------------------------------------------------------------------------
// <copyright file="TypesOfDrawn.cs" company="Whem">
//   Type of lottery enum class
// </copyright>
// <summary>
//   Defines the Enums type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryLib.Tools
{
    /// <summary>
    /// The enums.
    /// </summary>
    public partial class Enums
    {
        /// <summary>
        /// The types of drawn.
        /// </summary>
        public enum TypesOfDrawn
        {
            /// <summary>
            /// The by interval.
            /// </summary>
            ByInterval,

            /// <summary>
            /// The by occurrence.
            /// </summary>
            ByOccurrence,

            /// <summary>
            /// The by average steps.
            /// </summary>
            ByAverageSteps,

            /// <summary>
            /// The by average randoms.
            /// </summary>
            ByAverageRandoms,

            /// <summary>
            /// The by sums.
            /// </summary>
            BySums,

            /// <summary>
            /// The calculated.
            /// </summary>
            Calculated,

            /// <summary>
            /// The by distribution based current draw.
            /// </summary>
            ByDistributionBasedCurrentDraw,

            /// <summary>
            /// The by machine learning.
            /// </summary>
            ByMachineLearning,

            /// <summary>
            /// The all.
            /// </summary>
            All
        }
    }
}
