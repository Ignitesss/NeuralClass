using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> activationFunction;
        public static Func<double, double> activationFunctionDerivative;

        public int id;
        public double Output;
        public int layer;

        public double error;
        public double[] weightsToPrevLayer;

        public void setInput(double input)
        {
            if (layer == 0)
            {
                Output = input;
                return;
            }

            Output = activationFunction(input);
        }

        public Neuron(int id, int layer, int prevLayerCapacity, Random random)
        {
            this.id = id;
            this.layer = layer;
            this.error = 0;
            if (layer == -1)
            {
                Output = 1;
            }
            if (layer < 1)
            {
                weightsToPrevLayer = null;
            }
            else
            {
                weightsToPrevLayer = new double [prevLayerCapacity + 1];
                for (int i = 0; i < weightsToPrevLayer.Length; i++)
                {
                    weightsToPrevLayer[i] = random.NextDouble() * 2 - 1;
                }
            }
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        private const int hiddenLayersCount = 2;
        private const double learningRate = 0.1;

        private Neuron biasNeuron;
        private List<Neuron[]> layers;

        private Func<double[], double[], double> lossFunction;
        private Func<double, double, double> lossFunctionDerivative;

        public StudentNetwork(int[] structure)
        {
            if (structure.Length < 3)
            {
                throw new ArgumentException("Can not do 0 layers");
            }

            lossFunction = (output, aim) =>
            {
                double res = 0;
                for (int i = 0; i < aim.Length; i++)
                {
                    res += Math.Pow(aim[i] - output[i], 2);
                }

                return res * 0.5;
            };

            lossFunctionDerivative = (output, aim) => aim - output;

            Neuron.activationFunction = s => 1.0 / (1.0 + Math.Exp(-s));
            Neuron.activationFunctionDerivative = s => s * (1 - s);

            Random random = new Random();

            biasNeuron = new Neuron(0, -1, -1, random);
            int id = 1;

            layers = new List<Neuron[]>();

            for (int layer = 0; layer < structure.Length; layer++)
            {
                layers.Add(new Neuron[structure[layer]]);
                for (int i = 0; i < structure[layer]; i++)
                {
                    if (layer == 0)
                    {
                        layers[layer][i] = new Neuron(id, layer, -1, random);
                        continue;
                    }

                    layers[layer][i] = new Neuron(id, layer, structure[layer - 1], random);

                    id++;
                }
            }
        }


        public void forwardPropagation(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("Вы мне подсунули какой-то странный входной массив.");
            }
            for (int i = 0; i < layers[0].Length; i++)
            {
                layers[0][i].setInput(input[i]);
            }

            for (int layer = 1; layer < layers.Count; layer++)
            {
                for (int neuron = 0; neuron < layers[layer].Length; neuron++) 
                {
                    double scalar = 0;

                    for (int i = 0; i < layers[layer][neuron].weightsToPrevLayer.Length; i++)
                    {
                        if (i == 0)
                        {
                            scalar += biasNeuron.Output * layers[layer][neuron].weightsToPrevLayer[0];
                            continue;
                        }
                        scalar += layers[layer - 1][i - 1].Output * layers[layer][neuron].weightsToPrevLayer[i];
                    }
                    layers[layer][neuron].setInput(scalar);
                }
            }
        }

        public void backwardPropagation(Sample sample)
        {
            var aim = sample.outputVector;
            for (var i = 0; i < layers.Last().Length; i++)
            {
                layers.Last()[i].error = lossFunctionDerivative(layers.Last()[i].Output, aim[i]);
            }

            for (int layer = layers.Count - 1; layer >= 1; layer--)
            {
                foreach (var neuron in layers[layer])
                {
                    neuron.error *= Neuron.activationFunctionDerivative(neuron.Output);
                    for (int i = 0; i < neuron.weightsToPrevLayer.Length; i++)
                    {
                        if (i == 0)
                        {
                            biasNeuron.error += neuron.error * neuron.weightsToPrevLayer[0];
                            neuron.weightsToPrevLayer[0] += learningRate * neuron.error * biasNeuron.Output;
                            continue;
                        }

                        layers[layer - 1][i - 1].error += neuron.error * neuron.weightsToPrevLayer[i];
                        neuron.weightsToPrevLayer[i] += learningRate * neuron.error * layers[layer - 1][i - 1].Output;
                    }
                    neuron.error = 0;
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;
            while (true)
            {
                cnt++;
                forwardPropagation(sample.input);
                if (lossFunction(layers.Last().Select(n => n.Output).ToArray(), sample.outputVector) <=
                    acceptableError || cnt > 50)
                {
                    return cnt;
                }

                backwardPropagation(sample);
            }
        }

        double TrainOnSample(Sample sample, double acceptableError)
        {
            double loss;
            forwardPropagation(sample.input);
            loss = lossFunction(layers.Last().Select(n => n.Output).ToArray(), sample.outputVector);
            backwardPropagation(sample);
            return loss;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError,
            bool parallel)
        {
            var start = DateTime.Now;
            int totalSamplesCount = epochsCount * samplesSet.Count;
            int processedSamplesCount = 0;
            double sumError = 0;
            double mean;
            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                for (var index = 0; index < samplesSet.samples.Count; index++)
                {
                    var sample = samplesSet.samples[index];
                    sumError += TrainOnSample(sample, acceptableError);

                    processedSamplesCount++;
                    if (index % 100 == 0)
                    {
                        OnTrainProgress(1.0 * processedSamplesCount / totalSamplesCount,
                            sumError / (epoch * samplesSet.Count + index + 1), DateTime.Now - start);
                    }
                }

                mean = sumError / ((epoch + 1) * samplesSet.Count + 1);
                if (mean  <= acceptableError)
                {
                    OnTrainProgress(1.0,
                        mean, DateTime.Now - start);
                    return mean;
                }
            }
            mean = sumError / (epochsCount * samplesSet.Count + 1);
            OnTrainProgress(1.0,
                       mean, DateTime.Now - start);
            return sumError / (epochsCount * samplesSet.Count);
        }

        protected override double[] Compute(double[] input)
        {
            if (input.Length != layers[0].Length)
            {
                throw new ArgumentException("Too much data");
            }

            forwardPropagation(input);
            return layers.Last().Select(n => n.Output).ToArray();
        }
    }
}