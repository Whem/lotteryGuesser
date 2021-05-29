namespace LotteryGuesserXamarin.ViewModel
{
    using System;
    using System.Collections.Generic;
    using System.IO;

    using Google.Apis.Auth.OAuth2;
    using Google.Apis.Services;
    using Google.Apis.Sheets.v4;

    using LotteryLib.Tools;

    public class MainViewModel : BaseModel
    {
        public MainViewModel()
        {
           NavigationService.MenuViewChangeContentView(Enums.NavigationView.Login);
        }
    }
}
