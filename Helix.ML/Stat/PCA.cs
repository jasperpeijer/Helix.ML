using Helix.ML.LinAlg;

namespace Helix.ML.Stat;

/// <summary>
/// Principal Component Analysis (PCA) for dimensionality reduction.
/// </summary>
public class PCA
{
    /// <summary>
    /// The Principal Components (the new axes) extracted from the data.
    /// </summary>
    public Matrix Components { get; private set; }
    
    /// <summary>
    /// The center point of the training data.
    /// </summary>
    public double[] MeanVector { get; private set; }
    
    /// <summary>
    /// The percentage of total variance explained by each principal component.
    /// </summary>
    public double[] ExplainedVarianceRatio { get; private set; }

    /// <summary>
    /// Learns the principal components from the training data X.
    /// </summary>
    public void Fit(Matrix matrix, int maxIterations = 25000, double tolerance = 1e-7)
    {
        if (matrix.Rows < matrix.Cols)
            throw new ArgumentException("PCA typically requires more rows (samples) than columns (features).");

        MeanVector = matrix.ColumnMeans();
        var centeredData = matrix.MeanCenterColumns(MeanVector);
        var svd = centeredData.SVD(maxIterations, tolerance);
        Components = svd.V;
        ExplainedVarianceRatio = new double[matrix.Cols];
        double totalVariance = 0;

        for (var i = 0; i < matrix.Cols; i++)
        {
            var singularValue = svd.S[i, i];
            var variance = singularValue * singularValue;
            ExplainedVarianceRatio[i] = variance;
            totalVariance += variance;
        }

        for (var i = 0; i < matrix.Cols; i++)
        {
            ExplainedVarianceRatio[i] /= totalVariance;
        }
    }

    /// <summary>
    /// Projects new data onto the learned Principal Components.
    /// You can optionally specify how many dimensions to keep.
    /// </summary>
    public Matrix Transform(Matrix matrix, int? dimensionsToKeep = null)
    {
        if (MeanVector == null || Components.Rows == 0)
            throw new InvalidOperationException("You must call Fit() before calling Transform().");

        var keepCols = dimensionsToKeep ?? matrix.Cols;
        
        if (keepCols <= 0 || keepCols > matrix.Cols)
            throw new ArgumentOutOfRangeException(nameof(dimensionsToKeep), "Invalid number of dimensions.");

        var centeredData = matrix.MeanCenterColumns(MeanVector);
        var truncatedComponents = Components.ExtractColumns([..Enumerable.Range(0, keepCols)]);

        return centeredData * truncatedComponents;
    }

    /// <summary>
    /// The classic academic approach: Fits the model to the data and immediately 
    /// returns the dimensionality-reduced version of that exact same data.
    /// Ideal for exploratory data analysis and visualization.
    /// </summary>
    public Matrix FitTransform(Matrix matrix, int? dimensionsToKeep = null)
    {
        Fit(matrix);

        return Transform(matrix, dimensionsToKeep);
    }
}