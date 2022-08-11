namespace LotteryMaui.Entities
{
    public class LotteryUser
    {
        public int Id { get; set; }
        public string? UserName { get; set; }

        public string? PasswordHash { get; set; }
    }
}
