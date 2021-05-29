using System;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace LotteryGuesserXamarin
{
    using System.IO;

    using Google.Apis.Auth.OAuth2;
    using Google.Apis.Services;
    using Google.Apis.Sheets.v4;

    using LotteryGuesserXamarin.Services;
    using LotteryGuesserXamarin.View.Main;

    using Plugin.Settings;
    using Plugin.Settings.Abstractions;

    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            MainPage = new MainPage();
        }

        public static ISettings AppSettings => CrossSettings.Current;


        /// <summary>
        /// Gets the navigation service.
        /// </summary>
        public static ViewNavigationService NavigationService { get; } = new ViewNavigationService();

        public static CommonService CommonService { get; } = new CommonService();

        protected override void OnStart()
        {
        }

        protected override void OnSleep()
        {
        }

        protected override void OnResume()
        {
        }
    }
}
