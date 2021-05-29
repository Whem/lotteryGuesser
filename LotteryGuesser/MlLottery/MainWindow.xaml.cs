using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

namespace MlLottery
{
    using System.IO;
    using System.Net;

    using Encog.Engine.Network.Activation;
    using Encog.ML.Data.Basic;
    using Encog.Neural.Networks;
    using Encog.Neural.Networks.Layers;
    using Encog.Neural.Networks.Training.Propagation.Resilient;

    using LotteryLib.Model;
    using LotteryLib.Tools;

    using Path = System.Windows.Shapes.Path;

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            try
            {
                Lottery = new LotteryHandler(Enums.LotteryType.TheFiveNumberDraw, "Whem", false, false);
               
                LottoListResults dbl = null;
                if (CreateDatabase(Lottery.lotteryCollection, out dbl))
                {
                    var deep = 20;
                    var network = new BasicNetwork();
                    network.AddLayer(
                    new BasicLayer(null, true, 5 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 4 * 5 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 4 * 5 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationLinear(), true, 5));
                    network.Structure.FinalizeStructure();
                    var learningInput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        learningInput[i] = new double[deep * 5];
                        for (int j = 0, k = 0; j < deep; ++j)
                        {
                            var idx = 2 * deep - i - j;
                            LotteryResult data = dbl[idx];
                            learningInput[i][k++] = (double)data.V1;
                            learningInput[i][k++] = (double)data.V2;
                            learningInput[i][k++] = (double)data.V3;
                            learningInput[i][k++] = (double)data.V4;
                            learningInput[i][k++] = (double)data.V5;
                        }
                    }
                    var learningOutput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        learningOutput[i] = new double[5]
                        {
                            (double)data.V1,
                            (double)data.V2,
                            (double)data.V3,
                            (double)data.V4,
                            (double)data.V5
                        };
                    }
                    var trainingSet = new BasicMLDataSet(
                    learningInput,
                    learningOutput);

                    var train = new ResilientPropagation(
                    network, trainingSet);
                    train.NumThreads = Environment.ProcessorCount;
                    START:
                    network.Reset();
                    RETRY:
                    var step = 0;
                    do
                    {
                        train.Iteration();
                        Console.WriteLine("Train Error: {0}", train.Error);
                        ++step;
                    }
                    while (train.Error > 0.001 && step < 20);
                    var passedCount = 0;
                    for (var i = 0; i < deep; ++i)
                    {
                        var should =
                        new LotteryResult(learningOutput[i]);
                        var inputn = new BasicMLData(5 * deep);
                        Array.Copy(
                        learningInput[i],
                        inputn.Data,
                        inputn.Data.Length);
                        var comput =
                        new LotteryResult(
                        ((BasicMLData)network.
                        Compute(inputn)).Data);
                        var passed = should.ToString() == comput.ToString();
                        if (passed)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            ++passedCount;
                        }
                        else
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                        }
                        Console.WriteLine("{0} {1} {2} {3}",
                        should.ToString().PadLeft(17, ' '),
                        passed ? "==" : "!=",
                        comput.ToString().PadRight(17, ' '),
                        passed ? "PASS" : "FAIL");
                        Console.ResetColor();
                    }
                    var input = new BasicMLData(5 * deep);
                    for (int i = 0, k = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        input.Data[k++] = (double)data.V1;
                        input.Data[k++] = (double)data.V2;
                        input.Data[k++] = (double)data.V3;
                        input.Data[k++] = (double)data.V4;
                        input.Data[k++] = (double)data.V5;
                    }
                    var perfect = dbl[0];
                    var predict = new LotteryResult(
                    ((BasicMLData)network.Compute(input)).Data);
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("Predict: {0}", predict);
                    Console.ResetColor();
                    if (predict.IsOut())
                        goto START;

                    var t = passedCount < (deep * (double)9 / (double)10);
                    var isvalid = predict.IsValid();

                    if (t ||
                      !isvalid )
                        goto RETRY;
                    Console.WriteLine("Press any key for close...");
                    
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.ToString());
            }
            finally
            {
                
            }
        }

        public LotteryHandler Lottery { get; set; }

        static bool CreateDatabase(List<LotteryModel> fileDB, out LottoListResults dbl)
        {
            dbl = new LottoListResults();
            foreach (LotteryModel lotteryModel in fileDB)
            {
                var res = new LotteryResult(
                    lotteryModel.Numbers[0],
                    lotteryModel.Numbers[1],
                    lotteryModel.Numbers[2],
                    lotteryModel.Numbers[3],
                    lotteryModel.Numbers[4]
                );
                dbl.Add(res);
            }
            dbl.Reverse();
            return true;
        }

        class LottoListResults : List<LotteryResult> { }
    }
}
