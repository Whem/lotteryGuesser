using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryDesktopApp.ViewModels
{
    using System.Reactive;

    using LotteryLib.Model;
    using LotteryLib.Tools;

    using ReactiveUI;

    public class MainWindowViewModel : ViewModelBase, IScreen
    {
        // The Router associated with this Screen.
        // Required by the IScreen interface.
        public RoutingState Router { get; } = new RoutingState();

        // The command that navigates a user to first view model.
        public ReactiveCommand<Unit, IRoutableViewModel> GoNext { get; }

        // The command that navigates a user back.
        public ReactiveCommand<Unit, Unit> GoBack => Router.NavigateBack;

        public MainWindowViewModel()
        {
           // GoNext = ReactiveCommand.CreateFromObservable(() => Router.Navigate.Execute(new LoginViewModel(this)));

            Lottery = new LotteryHandler(Enums.LotteryType.TheFiveNumberDraw, "Whem", true, true);
            LotteryHandler.LotteryModelEvent += LotteryHandlerOnLotteryModelEvent;



            Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
            Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);


            Lottery.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
            Lottery.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);

            Lottery.SaveDataToGoogleSheet();
        }

        private void LotteryHandlerOnLotteryModelEvent(object sender, LotteryModel e)
        {
            
        }

        public LotteryHandler Lottery{ get; set; }
    }
}
