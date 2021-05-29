namespace LotteryGuesserXamarin.Services
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Reflection;
    using System.Threading.Tasks;

    using LotteryGuesserXamarin.Model.Tools;
    using LotteryGuesserXamarin.View.Contents;

    using LotteryLib.Tools;

    using ReactiveUI;

    using Xamarin.Forms;

    /// <summary>
    /// The view navigation service.
    /// </summary>
    public class ViewNavigationService : ReactiveObject, INavigationService
    {
        /// <summary>
        /// The _sync.
        /// </summary>
        private readonly object sync = new object();

        /// <summary>
        /// The _pages by key.
        /// </summary>
        private readonly Dictionary<string, Type> pagesByKey = new Dictionary<string, Type>();

        /// <summary>
        /// The navigation page stack.
        /// </summary>
        private readonly Stack<NavigationPage> navigationPageStack =
            new Stack<NavigationPage>();

        /// <summary>
        /// Gets the current page key.
        /// </summary>
        public string CurrentPageKey
        {
            get
            {
                lock (this.sync)
                {
                    if (this.CurrentNavigationPage?.CurrentPage == null)
                    {
                        return null;
                    }

                    var pageType = this.CurrentNavigationPage.CurrentPage.GetType();

                    return this.pagesByKey.ContainsValue(pageType)
                               ? this.pagesByKey.First(p => p.Value == pageType).Key
                               : null;
                }
            }
        }

        /// <summary>
        /// The current navigation page.
        /// </summary>
        private NavigationPage CurrentNavigationPage => this.navigationPageStack.Peek();

        /// <summary>
        /// The on set root page.
        /// </summary>
        /// <param name="rootPageKey">
        /// The root page key.
        /// </param>
        /// <returns>
        /// The <see cref="Page"/>.
        /// </returns>
        public Page OnSetRootPage(Enums.NavigationView rootPageKey)
        {
            var rootPage = this.OnGetPage(rootPageKey.ToString());
            this.navigationPageStack.Clear();
            var mainPage = new NavigationPage(rootPage);
            this.navigationPageStack.Push(mainPage);
            return mainPage;
        }

        public void OnConfigure(Enums.NavigationView pageKey)
        {
            Type tmpType = null;
            switch (pageKey)
            {
                case Enums.NavigationView.Login:
                    tmpType = typeof(LoginView);
                    break;
                case Enums.NavigationView.Lottery:
                    break;
                case Enums.NavigationView.Statistic:
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(pageKey), pageKey, null);
            }

            lock (this.sync)
            {
                if (this.pagesByKey.ContainsKey(pageKey.ToString()))
                {
                    this.pagesByKey[pageKey.ToString()] = tmpType;
                }
                else
                {
                    this.pagesByKey.Add(pageKey.ToString(), tmpType);
                }
            }
        }

        /// <summary>
        /// The go back.
        /// </summary>
        /// <returns>
        /// The <see cref="Task"/>.
        /// </returns>
        public async Task GoBack()
        {
            var navigationStack = this.CurrentNavigationPage.Navigation;
            if (navigationStack.NavigationStack.Count > 1)
            {
                await this.CurrentNavigationPage.PopAsync();
                return;
            }

            if (this.navigationPageStack.Count > 1)
            {
                this.navigationPageStack.Pop();
                await this.CurrentNavigationPage.Navigation.PopModalAsync();
                return;
            }

            await this.CurrentNavigationPage.PopAsync();
        }

        /// <summary>
        /// The navigate modal async.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        /// <param name="animated">
        /// The animated.
        /// </param>
        /// <returns>
        /// The <see cref="Task"/>.
        /// </returns>
        public async Task NavigateModalAsync(string pageKey, bool animated = true)
        {
            await this.NavigateModalAsync(pageKey, null, animated);
        }

        /// <summary>
        /// The navigate modal async.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        /// <param name="parameter">
        /// The parameter.
        /// </param>
        /// <param name="animated">
        /// The animated.
        /// </param>
        /// <returns>
        /// The <see cref="Task"/>.
        /// </returns>
        public async Task NavigateModalAsync(string pageKey, object parameter, bool animated = true)
        {
            var page = this.OnGetPage(pageKey, parameter);
            NavigationPage.SetHasNavigationBar(page, false);
            var modalNavigationPage = new NavigationPage(page);
            await this.CurrentNavigationPage.Navigation.PushModalAsync(modalNavigationPage, animated);
            this.navigationPageStack.Push(modalNavigationPage);
        }

        /// <summary>
        /// The navigate async.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        /// <param name="animated">
        /// The animated.
        /// </param>
        /// <returns>
        /// The <see cref="Task"/>.
        /// </returns>
        public async Task NavigateAsync(string pageKey, bool animated = true)
        {
            await this.NavigateAsync(pageKey, null, animated);
        }

        /// <summary>
        /// The navigate async.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        /// <param name="parameter">
        /// The parameter.
        /// </param>
        /// <param name="animated">
        /// The animated.
        /// </param>
        /// <returns>
        /// The <see cref="Task"/>.
        /// </returns>
        public async Task NavigateAsync(string pageKey, object parameter, bool animated = true)
        {
            var page = this.OnGetPage(pageKey, parameter);
            await this.CurrentNavigationPage.Navigation.PushAsync(page, animated);
        }

        /// <summary>
        /// The on get page.
        /// </summary>
        /// <param name="pageKey">
        /// The page key.
        /// </param>
        /// <param name="parameter">
        /// The parameter.
        /// </param>
        /// <returns>
        /// The <see cref="Page"/>.
        /// </returns>
        /// <exception cref="ArgumentException">
        /// </exception>
        /// <exception cref="InvalidOperationException">
        /// </exception>
        private Page OnGetPage(string pageKey, object parameter = null)
        {
            lock (this.sync)
            {
                if (!this.pagesByKey.ContainsKey(pageKey))
                {
                    throw new ArgumentException(
                        $"No such page: {pageKey}. Did you forget to call NavigationService.Configure?");
                }

                var type = this.pagesByKey[pageKey];
                ConstructorInfo constructor;
                object[] parameters;

                if (parameter == null)
                {
                    constructor = type.GetTypeInfo()
                        .DeclaredConstructors
                        .FirstOrDefault(c => !c.GetParameters().Any());

                    parameters = new object[]
                    {
                    };
                }
                else
                {
                    constructor = type.GetTypeInfo()
                        .DeclaredConstructors
                        .FirstOrDefault(
                            c =>
                            {
                                var p = c.GetParameters();
                                return p.Length == 1
                                       && p[0].ParameterType == parameter.GetType();
                            });

                    parameters = new[]
                    {
                    parameter
                };
                }

                if (constructor == null)
                {
                    throw new InvalidOperationException(
                        "No suitable constructor found for page " + pageKey);
                }

                var page = constructor.Invoke(parameters) as Page;
                return page;
            }
        }

        /// <summary>
        /// The main page content view.
        /// </summary>
        private ContentView mainPageContentView;

        /// <summary>
        /// Gets or sets the login page content view.
        /// </summary>
        public ContentView MainPageContentView
        {
            get => this.mainPageContentView;
            set => this.RaiseAndSetIfChanged(ref this.mainPageContentView, value);
        }

        /// <summary>
        /// The change content view.
        /// </summary>
        /// <param name="menusEnum">
        /// The menus enum.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// </exception>
        public void MenuViewChangeContentView(Enums.NavigationView menusEnum)
        {
            switch (menusEnum)
            {
                case Enums.NavigationView.Login:
                    MainPageContentView = new LoginView();
                    break;
                case Enums.NavigationView.Lottery:
                    MainPageContentView = new LotteryView();
                    break;
                case Enums.NavigationView.Statistic:
                    MainPageContentView = new StatisticView();
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(menusEnum), menusEnum, null);
            }
        }
    }
}