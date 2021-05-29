using System.Collections.Generic;
using System.Linq;

namespace LotteryLib.Model
{
    public class NumberSections
    {
       public int ActualNumber { get; set; }
        public List<SpecifyNumber> SpecifyNumberList{ get; set; }

        public NumberSections(int actualNumber)
        {
            ActualNumber = actualNumber;
            SpecifyNumberList = new List<SpecifyNumber>();

        }

        public void FindTheNextNumber(int nextNumber)
        {
            if(SpecifyNumberList.Count() == 0)
            {
                SpecifyNumberList.Add(new SpecifyNumber(nextNumber));
            }
            else
            {
                var getNextNumber = SpecifyNumberList.Where(x => x.NextNumber == nextNumber).FirstOrDefault();

                if(getNextNumber == null)
                {
                    SpecifyNumberList.Add(new SpecifyNumber(nextNumber));
                }
                else
                {
                    getNextNumber.IncreasPieces();
                }
            }
        }
    }
}
