using System.Numerics;
using LotteryMaui.Data;
using LotteryMaui.Model.Lottery.Model;
using LotteryMaui.Model.Lottery.Tools;
using LotteryMaui.SignalRHub;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;

namespace LotteryMaui.Controllers
{
    public class LotteryNumberController : Controller
    {
        private readonly DataContext _dataContext;
        private readonly LotteryHandler _lotteryHandler;


        public LotteryNumberController(DataContext dataContext, LotteryHandler lotteryHandler)
        {
            _dataContext = dataContext;
            _lotteryHandler = lotteryHandler;
        }

        [HttpPost("setLotterySettings")]
        public void SetLotterySettings(int lotteryType, int userId, bool isUserEarlierStatistics = true)
        {


           _lotteryHandler.LoadNeccessaryInformation((Enums.LotteryType) lotteryType, userId, isUserEarlierStatistics);
        }

        [HttpPost("generateLottery")]
        public void GenerateLottery(int lotteryDrawType, int generateType, int drawCount, int userId)
        {


            _lotteryHandler.CalculateNumbers(Enums.TypesOfDrawn.All, Enums.GenerateType.EachByEach, 2);
        }
    }
}
