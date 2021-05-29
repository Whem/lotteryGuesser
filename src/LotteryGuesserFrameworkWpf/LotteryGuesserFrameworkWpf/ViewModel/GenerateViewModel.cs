// --------------------------------------------------------------------------------------------------------------------
// <copyright file="GenerateViewModel.cs" company="Whem">
//   THis view contains lottery generating methods 
// </copyright>
// <summary>
//   Defines the GenerateViewModel type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryGuesserFrameworkWpf.ViewModel
{
    using GalaSoft.MvvmLight;

    using LotteryGuesserFrameworkWpf.Model.Services;

    /// <summary>
    /// The generate view model.
    /// </summary>
    public class GenerateViewModel : ViewModelBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GenerateViewModel"/> class.
        /// </summary>
        /// <param name="navigationService">
        /// The navigation service.
        /// </param>
        public GenerateViewModel(IFrameNavigationService navigationService)
        {
            this.NavigationService = navigationService;
        }

        /// <summary>
        /// Gets the navigation service.
        /// </summary>
        public IFrameNavigationService NavigationService { get; }
    }
}
