using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryDesktopApp.ViewModels
{
    using ReactiveUI;

    public class LoginViewModel : ViewModelBase, IRoutableViewModel
    {
        public string UrlPathSegment { get; } = Guid.NewGuid().ToString().Substring(0, 5);

        public IScreen HostScreen { get; }

        public LoginViewModel(IScreen screen)
        {
            HostScreen = screen;
        }

    }
}
