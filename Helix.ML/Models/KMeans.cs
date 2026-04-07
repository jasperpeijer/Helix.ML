using System.Data;
using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Stat;

namespace Helix.ML.Models;

public class KMeans
{
    public int Clusters { get; }
    public int MaxIterations { get; }
    public Matrix? Centroids { get; private set; }
    public double Inertia { get; private set; }

    public KMeans(int clusters, int maxIterations = 300)
    {
        if (clusters <= 0) 
            throw new ArgumentException("Number of clusters must be greater than zero.");
        
        Clusters = clusters;
        MaxIterations = maxIterations;
    }

    /// <summary>
    /// The Pipeline Bridge: Optionally encodes all non-numeric columns, optionally standardizes 
    /// numeric features, extracts the pure numeric matrix, and routes it to the math engine.
    /// </summary>
    public Column<string> FitPredict(DataFrame df, bool autoEncode = false, int seed = 42, 
        ScalerType scaler = ScalerType.None)
    {
        var trainingDf = autoEncode ? df.Clone().Encode() : df.Clone();

        if (scaler == ScalerType.Standardize)
        {
            foreach (var col in trainingDf.Columns.OfType<Column<double>>())
            {
                DataScaler.Standardize(col.Data);
            }
        }
        else if (scaler == ScalerType.MinMax)
        {
            foreach (var col in trainingDf.Columns.OfType<Column<double>>())
            {
                DataScaler.MinMaxScale(col.Data);
            }
        }

        var x = trainingDf.ToMatrix();

        return FitPredict(x, seed);
    }

    /// <summary>
    /// Pure, hardware-accelerated Euclidean clustering.
    /// </summary>
    public Column<string> FitPredict(Matrix matrix, int seed = 42)
    {
        var (rows, cols) = (matrix.Rows, matrix.Cols);
        
        if (Clusters > rows)
            throw new ArgumentException($"Cannot request {Clusters} clusters when the matrix only has {rows} rows.");
        
        var centroidsData = new double[Clusters * cols];
        var rand = new Random(seed);
        var initialIndices = Enumerable.Range(0, rows).OrderBy(x => rand.Next()).Take(Clusters).ToArray();

        for (var k = 0; k < Clusters; k++)
        {
            for (var j = 0; j < cols; j++)
            {
                centroidsData[k * cols + j] = matrix[initialIndices[k], j];
            }
        }

        var assignments = new int[rows];
        var changed = true;
        var iterations = 0;

        while (changed && iterations < MaxIterations)
        {
            changed = false;
            iterations++;
            var newAssignments = new int[rows];

            Parallel.For(0, rows, i =>
            {
                var minDistance = double.MaxValue;
                var bestCluster = 0;

                for (var k = 0; k < Clusters; k++)
                {
                    double distSq = 0;

                    for (var j = 0; j < cols; j++)
                    {
                        var diff = matrix[i, j] - centroidsData[k * cols + j];
                        distSq += diff * diff;
                    }

                    if (distSq < minDistance)
                    {
                        minDistance = distSq;
                        bestCluster = k;
                    }
                }
                
                newAssignments[i] = bestCluster;
            });

            for (var i = 0; i < rows; i++)
            {
                if (assignments[i] != newAssignments[i])
                {
                    changed = true;
                    assignments[i] = newAssignments[i];
                }
            }

            if (!changed) break;

            var newCentroidsData = new double[Clusters * cols];
            var counts = new int[Clusters];

            for (var i = 0; i < rows; i++)
            {
                var cluster = assignments[i];
                counts[cluster]++;

                for (var j = 0; j < cols; j++)
                {
                    newCentroidsData[cluster * cols + j] += matrix[i, j];
                }
            }

            for (var k = 0; k < Clusters; k++)
            {
                if (counts[k] > 0)
                {
                    for (var j = 0; j < cols; j++)
                    {
                        centroidsData[k * cols + j] = newCentroidsData[k * cols + j] / counts[k];
                    }
                }
            }
        }

        var totalInertia = 0.0;

        for (var i = 0; i < rows; i++)
        {
            var c = assignments[i];
            var distSq = 0.0;

            for (var j = 0; j < cols; j++)
            {
                var diff = matrix[i, j] - centroidsData[c * cols + j];
                distSq += diff * diff;
            }
            
            totalInertia += distSq;
        }

        Inertia = totalInertia;
        
        Centroids = new Matrix(Clusters, cols, centroidsData);

        return new Column<string>("Cluster", Array.ConvertAll(assignments, x => x.ToString()));
    }

    /// <summary>
    /// Predicts the cluster assignments for new, unseen data using the previously trained centroids.
    /// </summary>
    public Column<string> Predict(Matrix matrix)
    {
        if (Centroids == null)
            throw new InvalidOperationException("The model must be fitted before making predictions.");

        var trainedCentroids = Centroids.Value;
        
        if (matrix.Cols != Centroids?.Cols)
            throw new ArgumentException("The new data must have the exact same number of columns as the training data.");

        var rows = matrix.Rows;
        var assignments = new int[rows];

        Parallel.For(0, rows, (int i) =>
        {
            var minDistance = double.MaxValue;
            var bestCluster = 0;

            for (var k = 0; k < Clusters; k++)
            {
                double distSq = 0;

                for (var j = 0; j < matrix.Cols; j++)
                {
                    var diff = matrix[i, j] - trainedCentroids[k, j];
                    distSq += diff * diff;
                }

                if (distSq < minDistance)
                {
                    minDistance = distSq;
                    bestCluster = k;
                }
            }

            assignments[i] = bestCluster;
        });
        
        return new Column<string>("Predicted_Cluster", Array.ConvertAll(assignments, x => x.ToString()));
    }
}