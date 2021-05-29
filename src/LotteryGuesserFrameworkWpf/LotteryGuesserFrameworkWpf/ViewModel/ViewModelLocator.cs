/*
  In App.xaml:
  <Application.Resources>
      <vm:ViewModelLocator xmlns:vm="clr-namespace:LotteryGuesserFrameworkWpf"
                           x:Key="Locator" />
  </Application.Resources>
  
  In the View:
  DataContext="{Binding Source={StaticResource Locator}, Path=ViewModelName}"

  You can also use Blend to do all this with the tool's support.
  See http://www.galasoft.ch/mvvm
*/

using CommonServiceLocator;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Ioc;
using LotteryGuesserFrameworkWpf.Model.Services;
using System;

namespace LotteryGuesserFrameworkWpf.ViewModel
{
    /// <summary>
    /// This class contains static references to all the view models in the
    /// application and provides an entry point for the bindings.
    /// </summary>
    public class ViewModelLocator
    {
        public enum NavigationPages
        {
            Main,
            LotteryTypeSelection,
            LotteryGenerate,
            LotteryStatistic
        }


        /// <summary>
        /// Initializes a new instance of the ViewModelLocator class.
        /// </summary>
        public ViewModelLocator()
        {
            ServiceLocator.SetLocatorProvider(() => SimpleIoc.Default);

            ////if (ViewModelBase.IsInDesignModeStatic)
            ////{
            ////    // Create design time view services and models
            ////    SimpleIoc.Default.Register<IDataService, DesignDataService>();
            ////}
            ////else
            ////{
            ////    // Create run time view services and models
            ////    SimpleIoc.Default.Register<IDataService, DataService>();
            ////}
            this.SetupNavigation();

            SimpleIoc.Default.Register<MainViewModel>();
        }

        public MainViewModel Main
        {
            get
            {
                return ServiceLocator.Current.GetInstance<MainViewModel>();
            }
        }

        private void SetupNavigation()
        {
            var navigationService = new NavigationService();
            navigationService.Configure(NavigationPages.LotteryTypeSelection.ToString(), new Uri("../Views/LotterySelectionView.xaml", UriKind.Relative));
            navigationService.Configure(NavigationPages.LotteryGenerate.ToString(), new Uri("../Views/GenerateView.xaml", UriKind.Relative));
            navigationService.Configure(NavigationPages.LotteryStatistic.ToString(), new Uri("../Views/StatisticView.xaml", UriKind.Relative));
            SimpleIoc.Default.Register<IFrameNavigationService>(() => navigationService);
        }

        public static void Cleanup()
        {
            // TODO Clear the ViewModels
        }
    }
}