using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace LotteryCore.Model
{
    public class LotteryModel : ICloneable
    {
        private int _sum;
        private List<int> _numbers;

        public int Id { get; set; }

        public int Year { get; set; }

        public int WeekOfLotteryDrawing { get; set; }

        public DateTime DateOfDrawing { get; set; }

        public int FirstNumber { get; set; }

        public int SecondNumber { get; set; }

        public int ThirdNumber { get; set; }
        public int FourthNumber { get; set; }
        public int FifthNumber { get; set; }
        public List<string> XlsxString { get; set; }

        public SaveNumber.TypesOfDrawn Message { get; set; }

        public List<int> Numbers
        {
            get { return _numbers; }
            set
            {
                _numbers = value;
               
            }
        }

        public List<double> Avarages { get; set; }

        public List<int> StepBetweenNumbers { get; set; }

        public int Sum {
            get => _sum;
            set => _sum = value; }


        public List<int> RandomToGetNumber { get; set; }

        public LotteryModel(List<string> htmlString, int id)
        {
            Id = id;
            Numbers = new List<int>();
            RandomToGetNumber = new List<int>();
            XlsxString = htmlString;

            Year = Convert.ToInt16(htmlString[0]);
            WeekOfLotteryDrawing = Convert.ToInt16(htmlString[1]);

            DateOfDrawing = string.IsNullOrWhiteSpace(htmlString[2]) ? default(DateTime) : DateTime.Parse(htmlString[2]);

            
            FirstNumber = Convert.ToInt16(htmlString[11]);
            SecondNumber = Convert.ToInt16(htmlString[12]);
            ThirdNumber = Convert.ToInt16(htmlString[13]);
            FourthNumber = Convert.ToInt16(htmlString[14]);
            FifthNumber = Convert.ToInt16(htmlString[15]);

            Numbers.Add(FirstNumber);
            Numbers.Add(SecondNumber);
            Numbers.Add(ThirdNumber);
            Numbers.Add(FourthNumber);
            Numbers.Add(FifthNumber);

            
            GetSum();
            SetNumbers();
            GetAvareges();
            GetNumberWithRand();
        }

        public void AddNumber(int number)
        {
            Numbers.Add(number);
            GetSum();
            if (Numbers.Count == 5)
            {
                SetNumbers();
            }
        }

        public (bool, LotteryModel) ValidationTuple()
        {
            if (Numbers == null || Numbers.Count ==0) return (false,null);

            var duplicateKeys = Numbers.GroupBy(x => x)
                .Where(group => group.Count() > 1)
                .Select(group => group.Key);

            var hasOverRangedValue = Numbers.Where(x => x > 90 || x < 1);
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
                    if (rnd.Next(0, 91) == goalNumber)
                    {
                        RandomToGetNumber.Add(indexOfRandom);
                        break;
                    }
                    indexOfRandom++;
                }
            }
        }

        public LotteryModel()
        {
            Numbers = new List<int>();
        }

        public void SetNumbers()
        {
            FirstNumber = Numbers[0];
            SecondNumber = Numbers[1];
            ThirdNumber = Numbers[2];
            FourthNumber = Numbers[3];
            FifthNumber = Numbers[4];
            GetAvareges();
        }

        public void GetAvareges()
        {
            Avarages = new List<double>();
            StepBetweenNumbers = new List<int>();
            StepBetweenNumbers.Add(SecondNumber - FirstNumber);
            StepBetweenNumbers.Add(ThirdNumber - SecondNumber);
            StepBetweenNumbers.Add(FourthNumber - ThirdNumber);
            StepBetweenNumbers.Add(FifthNumber - FourthNumber);
            Avarages.Add((SecondNumber - FirstNumber) / 2);
            Avarages.Add((ThirdNumber - SecondNumber) / 2);
            Avarages.Add((FourthNumber - ThirdNumber) / 2);
            Avarages.Add((FifthNumber - FourthNumber) / 2);

            Avarages.Add((FirstNumber + SecondNumber + ThirdNumber + FourthNumber + FifthNumber) / 5);
            Avarages.Add(StepBetweenNumbers.Average());
        }

        public override string ToString()
        {
            //return string.Join(", ", new string[] { FirstNumber.ToString(), SecondNumber.ToString(), ThirdNumber.ToString(), FourthNumber.ToString(), FifthNumber.ToString() });
            return string.Join(", ", Numbers.OrderBy(x => x)) + " Sum: " + Sum;
        }

        public List<string> GetLotteryModelAsStrList()
        {
            
            List<string> concate = new List<string>
                {StatisticHandler.GetWeeksInYear().ToString(CultureInfo.InvariantCulture)};
            concate.AddRange(Numbers.OrderBy(x => x).Select(x => x.ToString()));
            concate.Add(Message.ToString());
            concate.Add(String.Join(',', Numbers.OrderBy(x => x).Select(x => x.ToString())));
            concate.Add("Whem");
            return concate;

        }



        public object Clone()
        {
            return this.MemberwiseClone();
        }
    }
}
