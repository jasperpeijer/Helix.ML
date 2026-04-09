using System.Data;
using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Stat;

namespace Helix.ML.Models;

public class LogisticRegression(double learningRate = .01, int maxIterations = 1000)
{
    public double[]? Weights { get; private set; }
    public double Bias { get; private set; }
    
    public double LearningRate { get; } = learningRate;
    public int MaxIterations { get; } = maxIterations;

    private ScalerType _scalerType = ScalerType.None;
    private Dictionary<string, (double Param1, double Param2)> _scalerParams = new();

    private string? _classZeroLabel, _classOneLabel;
    
    public ClassificationMetrics? TrainingMetrics { get; private set; }

    public void Fit(DataFrame df, string targetColumn, string positiveLabel, ScalerType scalerType = ScalerType.None,
        double evaluationThreshold = 0.5)
    {
        _scalerType = scalerType;
        var trainingDf = df.Clone();
        var yCol = trainingDf[targetColumn];
        trainingDf.Columns.Remove(yCol);

        var uniqueLabels = new HashSet<string>();

        for (var i = 0; i < yCol.Length; i++) uniqueLabels.Add(yCol.GetValue(i)?.ToString() ?? "");
        
        if (uniqueLabels.Count != 2)
            throw new InvalidOperationException($"Logistic Regression requires exactly 2 classes. Found {uniqueLabels.Count}.");
        
        if (!uniqueLabels.Contains(positiveLabel))
            throw new ArgumentException($"The positive label '{positiveLabel}' was not found in the target column.");

        _classOneLabel = positiveLabel;
        _classZeroLabel = uniqueLabels.First(l => l != positiveLabel);
        var yBinary = new double[yCol.Length];

        if (yCol is Column<string> strCol)
        {
            Parallel.For(0, strCol.Length, i => yBinary[i] = strCol[i] == _classOneLabel ? 1.0 : 0.0);
        }
        else if (yCol is Column<int> intCol)
        {
            var classOneInt = int.Parse(_classOneLabel);
            Parallel.For(0, intCol.Length, i => yBinary[i] = intCol[i] == classOneInt ? 1.0 : 0.0);
        }
        else if (yCol is Column<double> doubleCol)
        {
            var classOneDbl = double.Parse(_classOneLabel);
            Parallel.For(0, doubleCol.Length, i => yBinary[i] = doubleCol[i] == classOneDbl ? 1.0 : 0.0);
        }
        else
        {
            Parallel.For(0, yCol.Length, i => yBinary[i] = (yCol.GetValue(i)?.ToString() ?? "") == _classOneLabel ? 1.0 : 0.0);
        }

        if (_scalerType == ScalerType.Standardize)
        {
            foreach (var col in trainingDf.Columns.OfType<Column<double>>())
            {
                var (mean, _, stdDev) = DescriptiveStats.ComputeSummary(col.Data, asSample: false);
                _scalerParams[col.Name] = (mean, stdDev);

                Parallel.For(0, col.Data.Length, i =>
                {
                    col.Data[i] = stdDev == 0 ? 0 : (col.Data[i] - mean) / stdDev;
                });
            }
        }
        else if (_scalerType == ScalerType.MinMax)
        {
            foreach (var col in trainingDf.Columns.OfType<Column<double>>())
            {
                var min = col.Data.Min();
                var max = col.Data.Max();
                
                _scalerParams[col.Name] = (min, max);
                var range = max - min;

                Parallel.For(0, col.Data.Length, i =>
                {
                    col.Data[i] = range == 0 ? 0 : (col.Data[i] - min) / range;
                });
            }
        }
        
        Fit(trainingDf.ToMatrix(), yBinary, evaluationThreshold);
    }

    public void Fit(Matrix x, Matrix y, double evaluationThreshold = 0.5)
    {
        if (y.Rows == 1)
        {
            if (y.Cols != x.Rows) throw new ArgumentException("Y Matrix columns must match X Matrix rows.");
            var yArray = new double[y.Cols];
            Array.Copy(y.Data, 0, yArray, 0, yArray.Length);
            
            Fit(x, yArray);
        }
        else if (y.Cols == 1)
        {
            if (y.Rows != x.Rows) throw new ArgumentException("Y Matrix rows must match X Matrix rows.");
            var yArray = new double[y.Rows];
            Array.Copy(y.Data, 0, yArray, 0, yArray.Length);
            
            Fit(x, yArray, evaluationThreshold);
        }
        else
        {
            throw new ArgumentException("Y Matrix must be 1-Dimensional (either 1xN or Nx1).");
        }
    }

    public void Fit(Matrix x, double[] y, double evaluationThreshold = 0.5)
    {
        if (x.Rows != y.Length) throw new ArgumentException("Row mismatch between X and y.");
        
        var (rows, cols) = (x.Rows, x.Cols);
        Weights = new double[cols];
        Bias = 0.0;
        var weightGradients = new double[cols];
        
        for (var iter = 0; iter < maxIterations; iter++)
        {
            Array.Clear(weightGradients, 0, cols);
            var biasGradient = 0.0;

            for (var i = 0; i < rows; i++)
            {
                var linearModel = Bias;

                for (var j = 0; j < cols; j++)
                {
                    linearModel += x[i, j] * Weights[j];
                }

                var predictedProb = 1.0 / (1.0 + Math.Exp(-linearModel));
                var error = predictedProb - y[i];

                for (var j = 0; j < cols; j++)
                {
                    weightGradients[j] += error * x[i, j];
                }
                
                biasGradient += error;
            }

            for (var j = 0; j < cols; j++)
            {
                Weights[j] -= LearningRate * (weightGradients[j] / rows);
            }
            
            Bias -= LearningRate * (biasGradient / rows);
        }

        var rawProbabilities = PredictProbabilities(x);
        var mathPredictions = new double[rawProbabilities.Length];

        Parallel.For(0, rawProbabilities.Length, i =>
        {
            mathPredictions[i] = rawProbabilities[i] >= evaluationThreshold ? 1.0 : 0.0;
        });

        TrainingMetrics = ClassificationMetrics.Evaluate(y, mathPredictions, positiveValue: 1.0);
    }

    public Column<string> Predict(DataFrame df, double threshold = 0.5)
    {
        var testDf = df.Clone();
        var probabilities = PredictProbabilities(testDf);
        var stringPredictions = new string[probabilities.Length];

        Parallel.For(0, probabilities.Length, (int i) =>
        {
            stringPredictions[i] = probabilities[i] >= threshold ? _classOneLabel! : _classZeroLabel!;
        });
        
        return new Column<string>("Predicted_Class", stringPredictions);
    }
    
    public double[] PredictProbabilities(DataFrame df)
    {
        var testDf = df.Clone();

        if (_scalerType == ScalerType.Standardize)
        {
            foreach (var col in testDf.Columns.OfType<Column<double>>())
            {
                if (_scalerParams.TryGetValue(col.Name, out var stats))
                {
                    var (mean, stdDev) = stats;
                    Parallel.For(0, col.Data.Length, i =>
                    {
                        col.Data[i] = stdDev == 0 ? 0 : (col.Data[i] - mean) / stdDev;
                    });
                }
            }
        }
        else if (_scalerType == ScalerType.MinMax)
        {
            foreach (var col in testDf.Columns.OfType<Column<double>>())
            {
                if (_scalerParams.TryGetValue(col.Name, out var stats))
                {
                    var (min, max) = stats;
                    var range = max - min;
                    Parallel.For(0, col.Data.Length, i =>
                    {
                        col.Data[i] = range == 0 ? 0 : (col.Data[i] - min) / range;
                    });
                }
            }
        }
        
        return PredictProbabilities(testDf.ToMatrix());
    }

    public double[] PredictProbabilities(Matrix x)
    {
        if (Weights == null)
            throw new InvalidOperationException("Model must be fitted first.");

        var (rows, cols) = (x.Rows, x.Cols);
        var probabilities = new double[rows];

        Parallel.For(0, rows, i =>
        {
            var linearModel = Bias;

            for (var j = 0; j < cols; j++)
            {
                linearModel += x[i, j] * Weights[j];
            }
            
            probabilities[i] = 1.0 / (1.0 + Math.Exp(-linearModel));
        });

        return probabilities;
    }
}