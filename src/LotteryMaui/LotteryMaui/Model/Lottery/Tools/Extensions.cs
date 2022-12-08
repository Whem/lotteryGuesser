﻿// --------------------------------------------------------------------------------------------------------------------
// <copyright file="Extensions.cs" company="">
//   
// </copyright>
// <summary>
//   The extensions.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

using LotteryMaui.Model.Lottery.Model;

namespace LotteryMaui.Model.Lottery.Tools
{
    /// <summary>
    /// The extensions.
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// The clone.
        /// </summary>
        /// <param name="listToClone">
        /// The list to clone.
        /// </param>
        /// <typeparam name="T">
        /// </typeparam>
        /// <returns>
        /// The <see cref="IList"/>.
        /// </returns>
        public static IList<T> Clone<T>(this IList<T> listToClone) where T : ICloneable
        {
            return listToClone.Select(item => (T)item.Clone()).ToList();
        }

        public static void AddValueWithDetails(this List<LotteryModel> theList, LotteryModel lm, Enums.TypesOfDrawn tDrawn, bool isShowNumberOnConsole = true)
        {
            lm.Message = tDrawn;
            theList.Add(lm);

            if (isShowNumberOnConsole) Console.WriteLine(lm);
        }
        public static bool AddValueWithDetailsAndValidation(this List<LotteryModel> theList, (bool, LotteryModel) lm, Enums.TypesOfDrawn tDrawn, Enums.GenerateType generateType, bool isShowNumberOnConsole = true)
        {
            if(theList == null) theList = new List<LotteryModel>();
            if (!lm.Item1 || lm.Item2 == null) return false;
            if (!theList.Any(x => x.Numbers.SequenceEqual(lm.Item2.Numbers)))
            {
                lm.Item2.Message = tDrawn;
                lm.Item2.GenerateType = generateType;
                theList.Add(lm.Item2);

                if (isShowNumberOnConsole) Console.WriteLine(lm.Item2);
                
            }
            else
            {
                Console.WriteLine("Van hasonló a listában");
            }
            return true;
        }
    }
}