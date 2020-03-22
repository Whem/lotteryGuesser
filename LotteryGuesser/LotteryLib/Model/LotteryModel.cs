// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LotteryModel.cs" company="Whem">
//   Lottery
// </copyright>
// <summary>
//   Defines the LotteryModel type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryLib.Model
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LotteryLib.Tools;

    /// <summary>
    /// The lottery model.
    /// </summary>
    public class LotteryModel : ICloneable
    {
        /// <summary>
        /// The sum.
        /// </summary>
        private int sum;

        /// <summary>
        /// The numbers.
        /// </summary>
        private List<int> numbers;

        public int Id { get; set; }
        public LotteryRule LotteryRule { get; }

        public int Year { get; set; }

        public int WeekOfLotteryDrawing { get; set; }

        public DateTime DateOfDrawing { get; set; }

       
        public List<string> XlsxString { get; set; }

        public Enums.TypesOfDrawn Message { get; set; }

        public List<int> Numbers
        {
            get { return this.numbers; }
            set
            {
                this.numbers = value;
               
            }
        }

        public List<double> Avarages { get; set; }

        public List<int> StepBetweenNumbers { get; set; }

        public int Sum {
            get => this.sum;
            set => this.sum = value; }


        public List<int> RandomToGetNumber { get; set; }

        public LotteryModel(List<string> htmlString, int id, LotteryRule lotteryRule)
        {
            int skipItems = 0;

            switch (lotteryRule.LotteryType)
            {
                case Enums.LotteryType.TheSevenNumberDraw:
                case Enums.LotteryType.TheFiveNumberDraw:
                    skipItems = 11;
                    break;
                case Enums.LotteryType.TheSixNumberDraw:
                    skipItems = 13;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(lotteryRule.LotteryType), lotteryRule.LotteryType, null);
            }

            Id = id;
            LotteryRule = lotteryRule;
            Numbers = new List<int>();
            RandomToGetNumber = new List<int>();
            XlsxString = htmlString;

            Year = Convert.ToInt16(htmlString[0]);
            WeekOfLotteryDrawing = Convert.ToInt16(htmlString[1]);

            DateOfDrawing = string.IsNullOrWhiteSpace(htmlString[2]) ? default(DateTime) : DateTime.Parse(htmlString[2]);

            Numbers.AddRange(htmlString.Skip(skipItems).Take(lotteryRule.PiecesOfDrawNumber).ToList().Select(x => Convert.ToInt32(x)));

            GetSum();
            
            GetAvareges();
            GetNumberWithRand();
        }

        public void AddNumber(int number)
        {
            Numbers.Add(number);
            GetSum();
        }

        public (bool, LotteryModel) ValidationTuple()
        {
            if (Numbers == null || Numbers.Count ==0) return (false,null);

            var duplicateKeys = Numbers.GroupBy(x => x)
                .Where(group => group.Count() > 1)
                .Select(group => group.Key);

            var hasOverRangedValue = Numbers.Where(x => x > LotteryRule.MaxNumber || x < LotteryRule.MinNumber);
            return (!duplicateKeys.Any() && !hasOverRangedValue.Any(), this);
        }

        public void GetSum()
        {
            Sum = Numbers.Sum();
        }

        public void GetNumberWithRand()
        {
            foreach (var goalNumber in Numbers)
            {
                int indexOfRandom = 0;
                while (true)
                {

                    Random rnd = new Random();
                    if (rnd.Next(LotteryRule.MinNumber-1,LotteryRule.MaxNumber+1) == goalNumber)
                    {
                        RandomToGetNumber.Add(indexOfRandom);
                        break;
                    }
                    indexOfRandom++;
                }
            }
        }

        public LotteryModel(LotteryRule lotteryRule)
        {
            LotteryRule = lotteryRule;
            Numbers = new List<int>();
        }

        

        public void GetAvareges()
        {
            Avarages = new List<double>();
            StepBetweenNumbers = new List<int>();
            for (int i = 0; i < Numbers.Count-1; i++)
            {
                int overI = i + 1;
                var currentDif = Numbers[overI] - Numbers[i];

                StepBetweenNumbers.Add(currentDif);
                Avarages.Add(currentDif / 2);
            }
            Avarages.Add(Numbers.Average());
            Avarages.Add(StepBetweenNumbers.Average());
        }

        public override string ToString()
        {
            return string.Join(", ", Numbers.OrderBy(x => x));
        }

        public object Clone()
        {
            return this.MemberwiseClone();
        }
    }
}
