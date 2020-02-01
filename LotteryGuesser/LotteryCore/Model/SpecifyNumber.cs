using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LotteryGuesser.Model
{
    public class SpecifyNumber : ICloneable
    {
        public int NextNumber { get; set; }

        public int Pieces { get; set; }

        public SpecifyNumber(int nextNumber)
        {
            NextNumber = nextNumber;
        }

        public void IncreasPieces()
        {
            Pieces++;
        }

        public object Clone()
        {
            return this.MemberwiseClone();
        }

        public override string ToString()
        {
            return String.Join('-', new string[]{NextNumber.ToString(), Pieces.ToString()});
        }
    }
}
