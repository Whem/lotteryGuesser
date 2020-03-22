using System.Net;
using System.Net.Mail;

namespace LotteryLib.Model
{
    public class SmtpClientSend
    {
        public SmtpClientSend(string smtpServer, string userName, string password, int portNumber, string sendFrom, string sendTo, string body, string subject)
        {      
            SmtpClient client = new SmtpClient(smtpServer);
            client.Port = portNumber;
            client.EnableSsl = true;
            client.UseDefaultCredentials = false;
            client.Credentials = new NetworkCredential(userName, password);

            MailMessage mailMessage = new MailMessage();
            mailMessage.From = new MailAddress(sendFrom);
            mailMessage.To.Add(sendTo);
            
            mailMessage.Subject = subject;
            mailMessage.Body = body;
            client.Send(mailMessage);
            
           
        }

    }
}
