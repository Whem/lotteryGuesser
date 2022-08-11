// --------------------------------------------------------------------------------------------------------------------
// <copyright file="LotteryResult.cs" company="">
//   
// </copyright>
// <summary>
//   Defines the LotteryResult type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace LotteryMaui.Model.Lottery.Model
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
            this.V1 = v1;
            this.V2 = v2;
            this.V3 = v3;
            this.V4 = v4;
            this.V5 = v5;
           
        }
        public LotteryResult(double[] values)
        {
            this.V1 = (int)Math.Round(values[0]);
            this.V2 = (int)Math.Round(values[1]);
            this.V3 = (int)Math.Round(values[2]);
            this.V4 = (int)Math.Round(values[3]);
            this.V5 = (int)Math.Round(values[4]);
        }
        public bool IsValid()
        {
            return
                this.V1 >= 1 && this.V1 <= 90 &&
                this.V2 >= 1 && this.V2 <= 90 &&
                this.V3 >= 1 && this.V3 <= 90 &&
                this.V4 >= 1 && this.V4 <= 90 &&
                this.V5 >= 1 && this.V5 <= 90 &&
                this.V1 != this.V2 &&
                this.V1 != this.V3 &&
                this.V1 != this.V4 &&
                this.V1 != this.V5 &&
                this.V2 != this.V3 &&
                this.V2 != this.V4 &&
                this.V2 != this.V5 &&
                
                this.V3 != this.V4 &&
                this.V3 != this.V5 &&
                this.V4 != this.V5;
        }
        public bool IsOut()
        {
            return
                !(
                     this.V1 >= 1 && this.V1 <= 90 &&
                     this.V2 >= 1 && this.V2 <= 90 &&
                     this.V3 >= 1 && this.V3 <= 90 &&
                     this.V4 >= 1 && this.V4 <= 90 &&
                     this.V5 >= 1 && this.V5 <= 90);
        }
        public override string ToString()
        {
            return string.Format(
                "{0},{1},{2},{3},{4}", this.V1, this.V2, this.V3, this.V4, this.V5);
        }
    }
}
