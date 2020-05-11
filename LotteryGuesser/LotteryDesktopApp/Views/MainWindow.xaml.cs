using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace LotteryDesktopApp.Views
{
    using System.Reactive.Disposables;

    using Avalonia.ReactiveUI;

    using LotteryDesktopApp.ViewModels;

    using ReactiveUI;

    using Splat;

    public class MainWindow : ReactiveWindow<MainWindowViewModel>
    {
        public MainWindow()
        {
            Locator.CurrentMutable.InitializeReactiveUI();
            InitializeComponent();
            ViewModel = new MainWindowViewModel();
            
        }

        private void InitializeComponent()
        {
            this.WhenActivated(disposables => { });
            AvaloniaXamlLoader.Load(this);
        }
    }
}
