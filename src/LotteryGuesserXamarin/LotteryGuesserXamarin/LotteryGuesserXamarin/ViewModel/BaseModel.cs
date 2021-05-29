namespace LotteryGuesserXamarin.ViewModel
{
    using LotteryGuesserXamarin.Services;

    using ReactiveUI;

    /// <summary>
    /// The base model.
    /// </summary>
    public class BaseModel : ReactiveObject
    {
        /// <summary>
        /// The navigation service.
        /// </summary>
        public ViewNavigationService NavigationService => App.NavigationService;

        public CommonService CommonService => App.CommonService;
    }
}
