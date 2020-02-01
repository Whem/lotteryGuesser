using System;
using System.Collections.Generic;
using System.Text;

namespace LotteryCore.Model
{
    public class SaveNumbers
    {
        public int FirstNumber { get; set; }

        public int SecondNumber { get; set; }

        public int ThirdNumber { get; set; }
        public int FourthNumber { get; set; }
        public int FifthNumber { get; set; }

        public string Message { get; set; }

        public SaveNumbers(int[] numbers, string message)
        {
            FirstNumber = numbers[0];
            SecondNumber = numbers[1];
            ThirdNumber = numbers[2];
            FourthNumber = numbers[3];
            FifthNumber = numbers[4];
            Message = message;
        }
    }
}
