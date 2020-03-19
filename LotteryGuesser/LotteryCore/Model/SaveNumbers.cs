using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using LotteryCore.Tools;
using Newtonsoft.Json;

namespace LotteryCore.Model
{
    [JsonObject(MemberSerialization.OptIn)]
    public class SaveNumber
    {
        
        public int WeekOfPull { get; set; }
    
        public Enums.TypesOfDrawn Message { get; set; }
    
        public List<int> Numbers { get; set; }

        public List<double> DifferentInPercentage;

        public Enums.LotteryType LotteryType;

        public SaveNumber(string[] datas)
        {
            DifferentInPercentage = new List<double>();
            WeekOfPull = Convert.ToInt32(datas[0]);
            Numbers = datas[2].Split(',').Select(Int32.Parse).ToList();

            if (Enum.TryParse(datas[1], out Enums.TypesOfDrawn tDrawn))
            {
                Message = tDrawn;
            }
            if (Enum.TryParse(datas[4], out Enums.LotteryType lotteryType))
            {
                LotteryType = lotteryType;
            }




        }

        public override string ToString()
        {
            return string.Join(", ", Numbers.OrderBy(x => x));
        }
    }
}
