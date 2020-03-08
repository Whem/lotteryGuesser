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
        public enum TypesOfDrawn
        {
            ByIntervalForEachTimes,
            ByInterval,
            ByOccurrence,
            ByAverageSteps,
            ByAverageRandoms,
            BySums,
            Calculated,
            ByDistributionBasedCurrentDraw,
            Test
        }
        public int WeekOfPull { get; set; }
  
        public int FirstNumber { get; set; }
      
        public int SecondNumber { get; set; }
  
        public int ThirdNumber { get; set; }
      
        public int FourthNumber { get; set; }
     
        public int FifthNumber { get; set; }
    
        public TypesOfDrawn Message { get; set; }
    
        public List<int> Numbers { get; set; }

        public List<double> DifferentInPercentage;

        public SaveNumber(string[] datas)
        {
            DifferentInPercentage = new List<double>();
            WeekOfPull = Convert.ToInt32(datas[0]);
            Numbers = datas[7].Split(',').Select(Int32.Parse).ToList();
            if (!String.IsNullOrEmpty(datas[1]))
            {
                FirstNumber = Convert.ToInt32(datas[1]);
                SecondNumber = Convert.ToInt32(datas[2]);
                ThirdNumber = Convert.ToInt32(datas[3]);
                FourthNumber = Convert.ToInt32(datas[4]);
                FifthNumber = Convert.ToInt32(datas[5]);
            }
            else
            {
                FirstNumber = Numbers[0];
                SecondNumber = Numbers[1];
                ThirdNumber = Numbers[2];
                FourthNumber = Numbers[3];
                FifthNumber = Numbers[4];
            }

            if (Enum.TryParse(datas[6], out TypesOfDrawn tDrawn))
            {
                Message = tDrawn;
            }

            



        }

        public override string ToString()
        {
            return string.Join(", ", Numbers.OrderBy(x => x));
        }
    }
}
