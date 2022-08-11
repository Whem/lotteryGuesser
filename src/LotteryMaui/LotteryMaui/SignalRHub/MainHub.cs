using Microsoft.AspNetCore.SignalR;

namespace LotteryMaui.SignalRHub
{
    public class MainHub : Hub
    {
        public async IAsyncEnumerable<DateTime> Streaming(CancellationToken cancellationToken)
        {
            while (true)
            {
                yield return DateTime.UtcNow;
                await Task.Delay(1000, cancellationToken);
            }
        }

        public async void SendUpdateProduct(string name, string message, CancellationToken cancellationToken)
        {
            // Call the broadcastMessage method to update clients.
            await Clients.All.SendCoreAsync(name, new object?[]{message}, cancellationToken);
        }
    }
}
