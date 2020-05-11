using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace LotteryDesktopApp.Ucs
{
    public class LotteryUcs : UserControl
    {
        public LotteryUcs()
        {
            this.InitializeComponent();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
