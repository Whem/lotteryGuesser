using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MlLottery
{
    public class LotteryResult
    {
        public int V1 { get; private set; }
        public int V2 { get; private set; }
        public int V3 { get; private set; }
        public int V4 { get; private set; }
        public int V5 { get; private set; }
        
        public LotteryResult(int v1, int v2, int v3, int v4, int v5)
        {
            V1 = v1;
            V2 = v2;
            V3 = v3;
            V4 = v4;
            V5 = v5;
           
        }
        public LotteryResult(double[] values)
        {
            V1 = (int)Math.Round(values[0]);
            V2 = (int)Math.Round(values[1]);
            V3 = (int)Math.Round(values[2]);
            V4 = (int)Math.Round(values[3]);
            V5 = (int)Math.Round(values[4]);
        }
        public bool IsValid()
        {
            return
                V1 >= 1 && V1 <= 90 &&
                V2 >= 1 && V2 <= 90 &&
                V3 >= 1 && V3 <= 90 &&
                V4 >= 1 && V4 <= 90 &&
                V5 >= 1 && V5 <= 90 &&
                V1 != V2 &&
                V1 != V3 &&
                V1 != V4 &&
                V1 != V5 &&
                V2 != V3 &&
                V2 != V4 &&
                V2 != V5 &&
                
                V3 != V4 &&
                V3 != V5 &&
                V4 != V5;
        }
        public bool IsOut()
        {
            return
                !(
                     V1 >= 1 && V1 <= 90 &&
                     V2 >= 1 && V2 <= 90 &&
                     V3 >= 1 && V3 <= 90 &&
                     V4 >= 1 && V4 <= 90 &&
                     V5 >= 1 && V5 <= 90);
        }
        public override string ToString()
        {
            return string.Format(
                "{0},{1},{2},{3},{4}", V1, V2, V3, V4, V5);
        }
    }
}
