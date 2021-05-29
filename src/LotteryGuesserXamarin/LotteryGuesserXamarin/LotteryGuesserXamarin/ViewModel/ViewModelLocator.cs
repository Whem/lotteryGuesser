// --------------------------------------------------------------------------------------------------------------------
// <copyright file="ViewModelLocator.cs" company="Qx World Ltd.">
//   All rights reserved to the Qx World Ltd.
// </copyright>
// <summary>
//   Defines the ViewModelLocator type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryGuesserXamarin.ViewModel
{
    /// <summary>
    /// The view model locator.
    /// </summary>
    public static class ViewModelLocator
    {
        /// <summary>
        /// The menu view model.
        /// </summary>
        private static MainViewModel mainViewModel;

        /// <summary>
        /// The menu view model.
        /// </summary>
        public static MainViewModel Main =>
            mainViewModel ?? (mainViewModel = new MainViewModel());


        /// <summary>
        /// The menu view model.
        /// </summary>
        private static LoginViewModel loginViewModel;

        /// <summary>
        /// The menu view model.
        /// </summary>
        public static LoginViewModel Login =>
            loginViewModel ?? (loginViewModel = new LoginViewModel());


        /// <summary>
        /// The menu view model.
        /// </summary>
        private static LotteryViewModel lotteryViewModel;

        /// <summary>
        /// The menu view model.
        /// </summary>
        public static LotteryViewModel LotteryViewModel =>
            lotteryViewModel ?? (lotteryViewModel = new LotteryViewModel());
    }
}
