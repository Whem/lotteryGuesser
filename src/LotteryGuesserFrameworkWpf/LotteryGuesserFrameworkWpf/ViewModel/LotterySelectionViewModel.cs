using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LotteryGuesserFrameworkWpf.ViewModel
{
    using LotteryGuesserFrameworkWpf.Model.Services;

    public class LotterySelectionViewModel
    {
        private readonly IFrameNavigationService navigationService;

        public LotterySelectionViewModel(IFrameNavigationService navigationService)
        {
            this.navigationService = navigationService;
        }
    }
}
