using Helix.ML.Data;
using Helix.ML.LinAlg;

namespace Helix.ML.Models;

/// <summary>
/// An Ordinary Least Squares (OLS) Linear Regression model.
/// </summary>
public class LinearRegression
{
    public Matrix Weights { get; private set; }
    public double Bias { get; private set; }

    /// <summary>
    /// Trains the model directly from DataFrames.
    /// </summary>
    public void Fit(DataFrame x, DataFrame y)
    {
        Fit(x.ToMatrix(), y.ToMatrix());
    }
    
    /// <summary>
    /// Trains the model to map the input features (X) to the target outputs (y).
    /// </summary>
    public void Fit(Matrix x, Matrix y)
    {
        if (x.Rows != y.Rows)
            throw new ArgumentException("The number of rows in X must match the number of rows in y.");
        if (y.Cols != 1)
            throw new ArgumentException("The target matrix y must be a single column.");

        var ones = Matrix.Ones(x.Rows, 1);
        var xBias = ones | x;
        var beta = xBias.SolveLeastSquares(y);

        Bias = beta[0, 0];

        var weightsData = new double[x.Cols];

        for (var i = 0; i < x.Cols; i++)
        {
            weightsData[i] = beta[i + 1, 0];
        }
        
        Weights = new Matrix(x.Cols, 1, weightsData);
    }

    /// <summary>
    /// Predicts target values directly from a DataFrame.
    /// </summary>
    public Matrix Predict(DataFrame x)
    {
        return Predict(x.ToMatrix());
    }

    /// <summary>
    /// Predicts target values for the given input features.
    /// </summary>
    public Matrix Predict(Matrix x)
    {
        if (Weights.Data == null)
            throw new InvalidOperationException("The model must be fitted before making predictions.");

        return (x * Weights) + Bias;
    }

    /// <summary>
    /// Evaluates the model against a dataset and returns (MSE, R-Squared).
    /// </summary>
    public (double RMSE, double MSE, double R2) Evaluate(Matrix x, Matrix yActual)
    {
        var yPred = Predict(x);
        var n = yActual.Rows;
        
        // 1. Calculate Error (Residuals) using SIMD Matrix Subtraction
        var error = yActual - yPred;
        
        // 2. Sum of Squared Residuals (SSR) is the dot product of the error with itself
        var ssr = error.DotProduct(error);
        
        // 3. Total Sum of Squares (SST)
        // MeanCenterColumns shifts the target data so its average is 0, 
        // which allows us to find the variance using a single dot product.
        var yCentered = yActual.MeanCenterColumns();
        var sst = yCentered.DotProduct(yCentered);
        var mse = ssr / n;
        var r2 = sst == 0 ? 0 : 1.0 - (ssr / sst);

        return (Math.Sqrt(mse), mse, r2);
    }
}