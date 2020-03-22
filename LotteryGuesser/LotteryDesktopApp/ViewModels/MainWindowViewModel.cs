using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryDesktopApp.ViewModels
{
    using LotteryLib.Model;
    using LotteryLib.Tools;

    public class MainWindowViewModel : ViewModelBase
    {
        public string Greeting => "Welcome to Avalonia!";

        public MainWindowViewModel()
        {
            Lottery = new LotteryHandler(Enums.LotteryType.TheFiveNumberDraw, "Whem", true, true);



            Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
            Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);


            Lottery.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
            Lottery.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);

            //Lottery.SaveDataToGoogleSheet();
        }

        public LotteryHandler Lottery{ get; set; }
    }
}
