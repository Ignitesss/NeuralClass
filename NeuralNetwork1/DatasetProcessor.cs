using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace NeuralNetwork1
{
    using FastBitmap;
    public enum LetterType : byte { SH = 0, N, G, E, P, T, TS, Z, A, SOFT, Undef };
    public class DatasetProcessor
    {
        public static string LetterTypeToString(LetterType type)
        {
            switch (type)
            {
                case LetterType.SH:
                    return "Ш";
                case LetterType.N:
                    return "Н";
                case LetterType.G:
                    return "Г";
                case LetterType.E:
                    return "Е";
                case LetterType.SOFT:
                    return "Ь";
                case LetterType.Z:
                    return "З";
                case LetterType.T:
                    return "Т";
                case LetterType.TS:
                    return "Ц";
                case LetterType.P:
                    return "П";
                case LetterType.A:
                    return "А";
                case LetterType.Undef:
                    return "Неизвестно";
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, null);
            }
        }
        
        private const string databaseLocation = "..\\..\\dataset";
        private Random random;
        public int LetterCount { get; set; }

        private Dictionary<LetterType, List<string>> structure;
        public DatasetProcessor()
        {
            random = new Random();
            structure = new Dictionary<LetterType, List<string>>();
            structure[LetterType.E] = new List<string>();
            structure[LetterType.G] = new List<string>();
            structure[LetterType.N] = new List<string>();
            structure[LetterType.SOFT] = new List<string>();
            structure[LetterType.SH] = new List<string>();
            structure[LetterType.A] = new List<string>();
            structure[LetterType.P] = new List<string>();
            structure[LetterType.T] = new List<string>();
            structure[LetterType.Z] = new List<string>();
            structure[LetterType.TS] = new List<string>();
            foreach (var letter in structure)
            {
                DirectoryInfo d = new DirectoryInfo(databaseLocation + $"\\{LetterTypeToString(letter.Key)}");
                letter.Value.AddRange(d.GetFiles("*.jpeg").Select(f => f.FullName));
            }
        }
        public SamplesSet getTestDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            for (int type = 0; type < LetterCount; type++)
            {
                for (int i = 0; i < 100; i++)
                {
                    var sample = structure[(LetterType)type][random.Next(structure[(LetterType)type].Count)];
                    double[] input = new double[200];
                    using (FastBitmap fb = new FastBitmap(new Bitmap(sample)))
                    {
                        for (int x = 0; x < 200; x++)
                        {
                            for (int y = 0; y < 200; y++)
                            {
                                if (fb[x, y].ToArgb() != Color.White.ToArgb())
                                {
                                    input[x]++;
                                }
                            }
                        }
                    }
                    set.AddSample(new Sample(input, LetterCount, (LetterType)type));
                }
            }
            set.shuffle();
            return set;
        }
        
        public SamplesSet getTrainDataset(int count)
        {
            SamplesSet set = new SamplesSet();
            for (int type = 0; type < LetterCount; type++)
            {
                for (int i = 0; i < count/LetterCount; i++)
                {
                    var sample = structure[(LetterType)type][random.Next(structure[(LetterType)type].Count)];
                    double[] input = new double[200];
                    using (FastBitmap fb = new FastBitmap(new Bitmap(sample)))
                    {
                        for (int x = 0; x < 200; x++)
                        {
                            for (int y = 0; y < 200; y++)
                            {
                                if (fb[x, y].ToArgb() != Color.White.ToArgb())
                                {
                                    input[x]++;
                                }
                            }
                        }
                    }
                    set.AddSample(new Sample(input, LetterCount, (LetterType)type));
                }
            }
            set.shuffle();
            return set;
        }

        public Tuple<Sample, Bitmap> getSample()
        {
            var type = (LetterType) random.Next(LetterCount);
            var sample = structure[type][random.Next(structure[type].Count)];
            double[] input = new double[200];
            var bitmap = new Bitmap(sample);
            using (FastBitmap fb = new FastBitmap(bitmap))
            {
                for (int x = 0; x < 200; x++)
                {
                    for (int y = 0; y < 200; y++)
                    {
                        if (fb[x, y].ToArgb() != Color.White.ToArgb())
                        {
                            input[x]++;
                        }
                    }
                }
            }
            return Tuple.Create<Sample, Bitmap>(new Sample(input,LetterCount,type),bitmap);
        }

        public Sample getSample(Bitmap bitmap)
        {
            double[] input = new double[200];
            using (FastBitmap fb = new FastBitmap(bitmap))
            {
                for (int x = 0; x < 200; x++)
                {
                    for (int y = 0; y < 200; y++)
                    {
                        if (fb[x, y].ToArgb() != Color.White.ToArgb())
                        {
                            input[x]++;
                        }
                    }
                }
            }

            return new Sample(input, LetterCount);
        }
    }
}