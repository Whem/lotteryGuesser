using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryGuesserXamarin.ViewModel
{
    using System.Collections.ObjectModel;
    using System.Windows.Input;

    using LotteryLib.Model;
    using LotteryLib.Tools;

    using ReactiveUI;

    using Xamarin.Forms;

    public class LotteryViewModel : BaseModel
    {
        private ObservableCollection<LotteryModel> lotteryModels;

        private int countOfDrawn;

        public LotteryViewModel()
        {
            LotteryHandler.LotteryModelEvent += LotteryHandler_OnLotteryModelEvent;

            //Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
            //Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);


            //Lottery.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
            //Lottery.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);

            LotteryModels = new ObservableCollection<LotteryModel>();

            GenereateCommand = new Command((GenereateCommandExecute));

            SaveToGoogleSheet = new Command((SaveToGoogleSheetExecute));
        }

        private void SaveToGoogleSheetExecute()
        {
            CommonService.Lottery.SaveDataToGoogleSheet();
        }

        private void GenereateCommandExecute(object obj)
        {
            CommonService.Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
            CommonService.Lottery.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.GetTheBest, 1000);


            CommonService.Lottery.UseEarlierWeekPercentageForNumbersDraw(Enums.TypesOfDrawn.Calculated);
            CommonService.Lottery.CalculateNumbers(Enums.TypesOfDrawn.ByDistributionBasedCurrentDraw, Enums.GenerateType.Unique, 1);
        }

        private void LotteryHandler_OnLotteryModelEvent(object sender, LotteryModel e)
        {
            LotteryModels.Add(e);
        }

        public ObservableCollection<Enums.TypesOfDrawn> TypesOfDrawnsList { get; set; }

        public ObservableCollection<Enums.GenerateType> GenerateTypes { get; set; }

        public int CountOfDrawn
        {
            get => this.countOfDrawn;
            set => this.RaiseAndSetIfChanged(ref this.countOfDrawn, value);
        }

        public ObservableCollection<LotteryModel> LotteryModels
        {
            get => this.lotteryModels;
            set => this.RaiseAndSetIfChanged(ref this.lotteryModels, value);
        }

        public ICommand GenereateCommand { get; set; }

        public ICommand SaveToGoogleSheet { get; set; }
    }
}
