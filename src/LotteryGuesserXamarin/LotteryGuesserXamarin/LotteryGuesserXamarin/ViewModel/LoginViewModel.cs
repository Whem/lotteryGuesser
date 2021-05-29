// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LoginViewModel.cs" company="">
//   
// </copyright>
// <summary>
//   Defines the LoginViewModel type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryGuesserXamarin.ViewModel
{
    using System;
    using System.Collections.ObjectModel;
    using System.Linq;
    using System.Windows.Input;

    using LotteryLib.Model;
    using LotteryLib.Tools;

    using ReactiveUI;

    using Xamarin.Forms;

    public class LoginViewModel : BaseModel
    {
        private Enums.LotteryType selectedLotteryType;

        private bool isUseGoogleSheet;

        private bool isUseEarlierWeekDatas;

        public ObservableCollection<Enums.LotteryType> LotteryTypes { get; set; }


        public LoginViewModel()
        {
            LotteryTypes = new ObservableCollection<Enums.LotteryType>(Enum.GetValues(typeof(Enums.LotteryType)).Cast<Enums.LotteryType>());

            NextCommand = new Command((NextCommandExecute));

#if DEBUG
            SelectedLotteryType = Enums.LotteryType.TheFiveNumberDraw;
            CommonService.UserName = "Whem";
            IsUseEarlierWeekDatas = true;
            IsUseGoogleSheet = true;
#endif
        }

        public Enums.LotteryType SelectedLotteryType
        {
            get => this.selectedLotteryType;
            set => this.RaiseAndSetIfChanged(ref this.selectedLotteryType, value);
        }

        public bool IsUseGoogleSheet
        {
            get => this.isUseGoogleSheet;
            set => this.RaiseAndSetIfChanged(ref this.isUseGoogleSheet, value);
        }

        public bool IsUseEarlierWeekDatas
        {
            get => this.isUseEarlierWeekDatas;
            set => this.RaiseAndSetIfChanged(ref this.isUseEarlierWeekDatas, value);
        }

        private void NextCommandExecute()
        {
            CommonService.Lottery = new LotteryHandler(SelectedLotteryType, CommonService.UserName, IsUseGoogleSheet, IsUseEarlierWeekDatas);
            NavigationService.MenuViewChangeContentView(Enums.NavigationView.Lottery);
        }

        public ICommand NextCommand { get; set; }

    }
}
