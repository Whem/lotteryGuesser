using LotteryMaui.Entities;
using Microsoft.EntityFrameworkCore;

namespace LotteryMaui.Data
{
    public class DataContext : DbContext
    {
        public DataContext(DbContextOptions options) : base(options)
        {

        }

        public DbSet<LotteryUser>? LotteryUsers { get; set; }

        public DbSet<LotteryNumber>? LotteryNumbers { get; set; }


    }
}
