using LotteryMaui.Data;
using LotteryMaui.Entities;
using LotteryMaui.SignalRHub;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;

namespace LotteryMaui.Controllers
{
    public class AccountController : BaseApiController
    {
        public IHubContext<MainHub> HubContext { get; }
        private readonly DataContext _dataContext;
       

        public AccountController(DataContext dataContext, IHubContext<MainHub> hubContext)
        {
            HubContext = hubContext;
            _dataContext = dataContext;
            
        }

       

        [HttpGet("getUsers")]
        public Task<DbSet<LotteryUser>> GetUsers()
        {
            // await HubContext.Clients.All.SendCoreAsync("Test", new object?[] {"test"}, CancellationToken.None);
            if (_dataContext.LotteryUsers != null) return Task.FromResult(_dataContext.LotteryUsers);
            return Task.FromResult<DbSet<LotteryUser>>(null);
        }

        [HttpPost("loginUser")]
        public OkObjectResult LoginWithUser(string userName, string password)
        {
            

            var users = _dataContext.LotteryUsers;

            if (users != null)
            {
                var user = users.FirstOrDefault(x => x.UserName == userName );
                if (user == null) return new OkObjectResult(NoContent());
                if(user.PasswordHash == new PasswordHasher<object?>().HashPassword(null, password)) return Ok(user);
            }

            return new OkObjectResult(NoContent());
        }

        [HttpPost("registerUser")]
        public async Task<OkResult> RegisterUser(string userName, string password)
        {


            var users = _dataContext.LotteryUsers;

            if (users != null)
            {
                var user = users.FirstOrDefault(x => x.UserName == userName);
                if (user == null) return null;
            }

            LotteryUser newUser = new LotteryUser
            {
                UserName = userName,
                PasswordHash = new PasswordHasher<object?>().HashPassword(null, password)
            };
            if (_dataContext.LotteryUsers != null) await _dataContext.LotteryUsers.AddAsync(newUser);

            return Ok();
        }
    }
}
