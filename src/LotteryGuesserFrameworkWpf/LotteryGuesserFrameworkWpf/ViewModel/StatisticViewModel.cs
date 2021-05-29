namespace LotteryGuesserFrameworkWpf.ViewModel
{
    using LotteryGuesserFrameworkWpf.Model.Services;

    public class StatisticViewModel
    {
        private readonly IFrameNavigationService navigationService;

        public StatisticViewModel(IFrameNavigationService navigationService)
        {
            this.navigationService = navigationService;
        }
    }
}
