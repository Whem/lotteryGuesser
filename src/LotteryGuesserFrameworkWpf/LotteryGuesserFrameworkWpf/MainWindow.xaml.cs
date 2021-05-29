using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using LotteryLib.Model;
using LotteryLib.Tools;

namespace LotteryGuesserFrameworkWpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        //Lottery = new LotteryHandler(Enums.LotteryType.TheFiveNumberDraw, "Whem", true, true);
        //LotteryHandler.LotteryModelEvent += LotteryHandlerOnLotteryModelEvent;



        //Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
        //Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);


        //Lottery.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
        //Lottery.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);
        //Lottery.SaveDataToGoogleSheet();
    }
}
