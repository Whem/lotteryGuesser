using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

namespace LotteryCore.Model
{
    [JsonObject(MemberSerialization.OptIn)]
    public class SaveNumber
    {
        [JsonProperty]
        public int WeekOfPull { get; set; }
        [JsonProperty]
        public int FirstNumber { get; set; }
        [JsonProperty]
        public int SecondNumber { get; set; }
        [JsonProperty]
        public int ThirdNumber { get; set; }
        [JsonProperty]
        public int FourthNumber { get; set; }
        [JsonProperty]
        public int FifthNumber { get; set; }
        [JsonProperty]
        public string Message { get; set; }
        [JsonProperty]
        public List<int> Numbers { get; set; }

        public List<double> DifferentInPercentage;

        public SaveNumber()
        {
            DifferentInPercentage = new List<double>();
        }

        public SaveNumber(int[] numbers, string message,int weekOfPull =0, bool generateAutomaticCurrentWeek = true)
        {
            Numbers = numbers.ToList();
            FirstNumber = numbers[0];
            SecondNumber = numbers[1];
            ThirdNumber = numbers[2];
            FourthNumber = numbers[3];
            FifthNumber = numbers[4];
            Message = message;
            WeekOfPull = generateAutomaticCurrentWeek ? StatisticHandler.GetWeeksInYear() : weekOfPull;

        }

        public override string ToString()
        {
            return string.Join(", ", Numbers.OrderBy(x => x));
        }
    }
}
