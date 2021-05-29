namespace LotteryGuesserXamarin.Model.Tools
{
    using System.Threading.Tasks;

    using LotteryLib.Tools;

    /// <summary>
    /// The NavigationService interface.
    /// </summary>
    public interface INavigationService 
    {
        string CurrentPageKey { get; }

        /// <summary>
        /// The on configure.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        void OnConfigure(Enums.NavigationView pageKey);

        Task GoBack();

        Task NavigateModalAsync(string pageKey, bool animated = true);

        Task NavigateModalAsync(string pageKey, object parameter, bool animated = true);

        Task NavigateAsync(string pageKey, bool animated = true);

        Task NavigateAsync(string pageKey, object parameter, bool animated = true);
    }
}