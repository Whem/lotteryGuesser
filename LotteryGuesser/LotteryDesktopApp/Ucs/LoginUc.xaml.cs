// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LoginUc.xaml.cs" company="">
//   
// </copyright>
// <summary>
//   The login uc.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryDesktopApp.Ucs
{
    using Avalonia.Markup.Xaml;
    using Avalonia.ReactiveUI;

    using LotteryDesktopApp.ViewModels;

    using ReactiveUI;

    /// <summary>
    /// The login uc.
    /// </summary>
    public class LoginUc : ReactiveUserControl<LoginViewModel>
    {
        public LoginUc()
        {
            this.InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.WhenActivated(disposables => { });
            AvaloniaXamlLoader.Load(this);
        }
    }
}
