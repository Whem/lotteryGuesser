﻿using System.Collections.Generic;

namespace LotteryLib.Model
{
    public class IntervallNumber
    {
        public int StartInterVal { get; set; }
        public int StopInterval { get; set; }

        public List<int> ActualNumberList { get; set; }

        public List<int> AfterNumberList { get; set; }

        public IntervallNumber(int startInterVal, int stopInterval)
        {
            StartInterVal = startInterVal;
            StopInterval = stopInterval;
            ActualNumberList = new List<int>();
            AfterNumberList = new List<int>();
        }
    }
}
