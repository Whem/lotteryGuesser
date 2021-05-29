using GalaSoft.MvvmLight;

namespace LotteryGuesserFrameworkWpf.ViewModel
{
    using GalaSoft.MvvmLight.Command;
    using LotteryGuesserFrameworkWpf.Model.Services;

    /// <summary>
    /// This class contains properties that the main View can data bind to.
    /// <para>
    /// Use the <strong>mvvminpc</strong> snippet to add bindable properties to this ViewModel.
    /// </para>
    /// <para>
    /// You can also use Blend to data bind with the tool's support.
    /// </para>
    /// <para>
    /// See http://www.galasoft.ch/mvvm
    /// </para>
    /// </summary>
    public class MainViewModel : ViewModelBase
    {
        private IFrameNavigationService _navigationService;

        private RelayCommand _loadedCommand;

        public RelayCommand LoadedCommand
        {
            get
            {
                return _loadedCommand
                       ?? (_loadedCommand = new RelayCommand(
                               () =>
                                   {
                                       _navigationService.NavigateTo(ViewModelLocator.NavigationPages.LotteryTypeSelection.ToString());
                                   }));
            }
        }

        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public MainViewModel(IFrameNavigationService navigationService)
        {
            this._navigationService = navigationService;
        }
    }
}