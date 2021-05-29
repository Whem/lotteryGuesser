using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryGuesserXamarin.Services
{
    using LotteryLib.Model;

    using ReactiveUI;

    public class CommonService : ReactiveObject
    {
        private string userName;

        private LotteryHandler lottery;

        public string UserName
        {
            get => this.userName;
            set => this.RaiseAndSetIfChanged(ref this.userName, value);
        }

        public LotteryHandler Lottery
        {
            get => this.lottery;
            set => this.lottery = value;
        }
    }
}
