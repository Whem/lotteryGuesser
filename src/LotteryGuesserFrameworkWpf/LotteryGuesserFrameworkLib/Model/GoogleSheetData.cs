using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using Google.Apis.Auth.OAuth2;
using Google.Apis.Services;
using Google.Apis.Sheets.v4;
using Google.Apis.Sheets.v4.Data;
using Google.Apis.Util.Store;
using Newtonsoft.Json;

namespace LotteryLib.Model
{
    using System.Collections.ObjectModel;

    public class GoogleSheetData
    {
        public string UserName { get; }
        List<string[]> datas;
        SheetsService service;
        String spreadsheetId = "1rPiOnFUYxrvZT8XT27n1cxSvsr5NBtRMDnhUH5LLbXM";
        String range = "Numbers!A2:I";

        // If modifying these scopes, delete your previously saved credentials
        // at ~/.credentials/sheets.googleapis.com-dotnet-quickstart.json
        static string[] Scopes = { SheetsService.Scope.Spreadsheets };
        static string ApplicationName = "LotteryGuesser";
        public GoogleSheetData(string userName)
        {
            UserName = userName;


            UserCredential credential;

            using (var stream =
                new FileStream("credentials.json", FileMode.Open, FileAccess.Read))
            {
                // The file token.json stores the user's access and refresh tokens, and is created
                // automatically when the authorization flow completes for the first time.
                string credPath = "token.json";
                credential = GoogleWebAuthorizationBroker.AuthorizeAsync(
                    GoogleClientSecrets.Load(stream).Secrets,
                    Scopes,
                    "user",
                    CancellationToken.None,
                    new FileDataStore(credPath, true)).Result;
                Console.WriteLine("Credential file saved to: " + credPath);
            }

            // Create Google Sheets API service.
            
            service = new SheetsService(new BaseClientService.Initializer()
            {
                HttpClientInitializer = credential,
                ApplicationName = ApplicationName,
                
            });

           
            SpreadsheetsResource.ValuesResource.GetRequest request =
                    service.Spreadsheets.Values.Get(spreadsheetId, range);

            // Prints the names and majors of students in a sample spreadsheet:
            // https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
            ValueRange response = request.Execute();
            var data = response.Values;
            datas = new List<string[]>();
            if (data != null && data.Count > 0)
            {

                foreach (var row in data)
                {
                    datas.Add(row.Select(s => (string)s).ToArray());
                }
            }
            else
            {
                Console.WriteLine("No data found.");
            }
           
           
        }
        public List<string> GetLotteryModelAsStrList(LotteryModel lm)
        {

            List<string> concate = new List<string>
                {LotteryHandler.GetWeeksInYear().ToString(CultureInfo.InvariantCulture)};
            concate.Add(lm.Message.ToString());
            concate.Add(String.Join(",", lm.Numbers.OrderBy(x => x).Select(x => x.ToString())));
            concate.Add(UserName);
            concate.Add(lm.LotteryRule.LotteryType.ToString());
            return concate;

        }

        /// <summary>
        /// The save numbers to sheet.
        /// </summary>
        /// <param name="lotteryModels">
        /// The lottery models.
        /// </param>
        /// <returns>
        /// The <see cref="AppendValuesResponse"/>.
        /// </returns>
        public AppendValuesResponse SaveNumbersToSheet(List<LotteryModel> lotteryModels)
        {
            IList<IList<object>> values = new List<IList<object>>();
            foreach (var lotteryModel in lotteryModels)
            {
                values.Add(GetLotteryModelAsStrList(lotteryModel).Cast<object>().ToList());
            }

            // How the input data should be interpreted.
            SpreadsheetsResource.ValuesResource.AppendRequest.ValueInputOptionEnum valueInputOption = SpreadsheetsResource.ValuesResource.AppendRequest.ValueInputOptionEnum.USERENTERED;  // TODO: Update placeholder value.

            // How the input data should be inserted.
            SpreadsheetsResource.ValuesResource.AppendRequest.InsertDataOptionEnum insertDataOption = SpreadsheetsResource.ValuesResource.AppendRequest.InsertDataOptionEnum.OVERWRITE;  // TODO: Update placeholder value.

            // TODO: Assign values to desired properties of `requestBody`:
            var requestBody = new ValueRange() { Values = values };

            var request = service.Spreadsheets.Values.Append(requestBody, spreadsheetId, range);
            request.ValueInputOption = valueInputOption;
            request.InsertDataOption = insertDataOption;

            // To execute asynchronously in an async method, replace `request.Execute()` as shown:
            var response = request.Execute();
           
            // TODO: Change code below to process the `response` object:
            Console.WriteLine(JsonConvert.SerializeObject(response));

            return response;
        }

        /// <summary>
        /// The get data.
        /// </summary>
        /// <returns>
        /// The <see cref="List"/>.
        /// </returns>
        public List<string[]> GetData()
        {
            return datas;
        }
    }
}
