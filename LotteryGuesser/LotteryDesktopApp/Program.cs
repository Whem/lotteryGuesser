using System;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Logging.Serilog;
using Avalonia.ReactiveUI;

namespace LotteryDesktopApp
{
    using Avalonia.Threading;

    using LotteryDesktopApp.Ucs;
    using LotteryDesktopApp.ViewModels;

    using ReactiveUI;

    using Splat;

    class Program
    {
        // Initialization code. Don't use any Avalonia, third-party APIs or any
        // SynchronizationContext-reliant code before AppMain is called: things aren't initialized
        // yet and stuff might break.
        public static void Main(string[] args) => BuildAvaloniaApp()
            .StartWithClassicDesktopLifetime(args);

        // Avalonia configuration, don't remove; also used by visual designer.
        public static AppBuilder BuildAvaloniaApp()
        {
            // Router uses Splat.Locator to resolve views for view models, so we need to register our views.
            
            Locator.CurrentMutable.Register(() => new LoginUc(), typeof(IViewFor<LoginViewModel>));

            RxApp.MainThreadScheduler = AvaloniaScheduler.Instance;

            return AppBuilder
                .Configure<App>()
                .UseReactiveUI()
                .UsePlatformDetect()
                .LogToDebug();
        }
    }
}
