using LotteryMaui.Model.Lottery.Tools;

namespace LotteryMaui.Entities
{
    public class LotteryNumber
    {
        public int Id { get; set; }

        public int LotteryDrawId { get; set; }

        public int WeekOfPull { get; set; }

        public Enums.TypesOfDrawn Message { get; set; }

        public string Numbers { get; set; }

        public Enums.LotteryType LotteryType { get; set; }

        //public LotteryNumber(string[] datas)
        //{
        //    DifferentInPercentage = new List<double>();
        //    WeekOfPull = Convert.ToInt32(datas[0]);
        //    Numbers = datas[2].Split(',').Select(Int32.Parse).ToList();

        //    if (Enum.TryParse(datas[1], out Enums.TypesOfDrawn tDrawn))
        //    {
        //        Message = tDrawn;
        //    }
        //    if (Enum.TryParse(datas[4], out Enums.LotteryType lotteryType))
        //    {
        //        LotteryType = lotteryType;
        //    }




        //}

        public override string ToString()
        {
            return string.Join(", ", Numbers.OrderBy(x => x));
        }
    }
}
