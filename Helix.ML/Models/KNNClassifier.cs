using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Stat;

namespace Helix.ML.Models;

/// <summary>
/// A high-performance K-Nearest Neighbors classifier using Euclidean distance and spatial voting.
/// </summary>
public class KNNClassifier
{
    public int K { get; }

    private Matrix? _xTrain;
    private string[] _yTrain;

    private ScalerType? _scalerType = ScalerType.None;
    private Dictionary<string, (double Param1, double Param2)> _scalerParams = new();
    
    public KNNClassifier(int k = 3)
    {
        if (k <= 0)
            throw new ArgumentException("K must be an integer greater than zero.");
        
        K = k;
    }

    public void Fit(DataFrame df, string targetColumn, ScalerType scaler = ScalerType.None)
    {
        _scalerType = scaler;
        var trainingDf = df.Clone();
        var yCol = trainingDf.GetColumn<string>(targetColumn);

        trainingDf.Columns.Remove(yCol);

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
        
        Fit(trainingDf.ToMatrix(), yCol);
    }

    public void Fit(Matrix x, Column<string> y)
    {
        if (x.Rows != y.Length)
            throw new ArgumentException($"Row mismatch: X has {x.Rows} rows, but y has {y.Length} labels.");
        
        _xTrain = x;
        _yTrain = new string[y.Length];

        for (var i = 0; i < y.Length; i++)
        {
            _yTrain[i] = y[i];
        }
    }

    public Column<string> Predict(DataFrame df)
    {
        var dfCopy = df.Clone();
        
        if (_scalerType == ScalerType.Standardize)
        {
            foreach (var col in dfCopy.Columns.OfType<Column<double>>())
            {
                if (_scalerParams.TryGetValue(col.Name, out var stats))
                {
                    Parallel.For(0, col.Data.Length, i =>
                    {
                        var (mean, stdDev) = stats;
                        col.Data[i] = stdDev == 0 ? 0 : (col.Data[i] - mean) / stdDev;
                    });
                }
            }
        }
        else if (_scalerType == ScalerType.MinMax)
        {
            foreach (var col in dfCopy.Columns.OfType<Column<double>>())
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

        return Predict(dfCopy.ToMatrix());
    }

    public Column<string> Predict(Matrix x)
    {
        if (_xTrain == null || _yTrain == null)
            throw new InvalidOperationException("The model must be fitted with training data before predicting.");

        var trainX = _xTrain.Value;
        
        if (x.Cols != trainX.Cols)
            throw new ArgumentException($"The new data has {x.Cols} columns, but the model memorized {trainX.Cols} columns.");

        var newRows = x.Rows;
        var trainRows = trainX.Rows;
        var cols = x.Cols;
        var predictions = new string[newRows];

        Parallel.For(0, newRows, i =>
        {
            var distances = new (double Distance, int TrainIndex)[trainRows];

            for (var t = 0; t < trainRows; t++)
            {
                double distSq = 0;

                for (var j = 0; j < cols; j++)
                {
                    var diff = x[i, j] - trainX[t, j];
                    distSq += diff * diff;
                }

                distances[t] = (distSq, t);
            }
            
            Array.Sort(distances, (a, b) => a.Distance.CompareTo(b.Distance));

            var voteCounts = new Dictionary<string, int>();

            for (var k = 0; k < K; k++)
            {
                var neighborLabel = _yTrain[distances[k].TrainIndex];
                
                if (!voteCounts.ContainsKey(neighborLabel))
                    voteCounts[neighborLabel] = 0;
                
                voteCounts[neighborLabel]++;
            }

            var bestLabel = "";
            var maxVotes = -1;

            foreach (var kvp in voteCounts)
            {
                if (kvp.Value > maxVotes)
                {
                    maxVotes = kvp.Value;
                    bestLabel = kvp.Key;
                }
            }
            
            predictions[i] = bestLabel;
        });

        return new Column<string>("Predicted_Class", predictions);
    }
}