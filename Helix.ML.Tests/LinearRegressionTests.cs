using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Models;

namespace Helix.ML.Tests;

public class LinearRegressionTests
{
    [Fact]
    public void Fit_CalculatesCorrectWeightsAndBias()
    {
        // Arrange: y = (2 * x1) + (3 * x2) + 10
        var X = new Matrix(3, 2, [
            1, 1,
            2, 2,
            3, 1
        ]);
        var y = new Matrix(3, 1, [
            15, // 2(1) + 3(1) + 10
            20, // 2(2) + 3(2) + 10
            19  // 2(3) + 3(1) + 10
        ]);

        var model = new LinearRegression();

        // Act
        model.Fit(X, y);

        // Assert
        Assert.True(Math.Abs(10.0 - model.Bias) < 1e-10);
        Assert.True(Math.Abs(2.0 - model.Weights[0, 0]) < 1e-10);
        Assert.True(Math.Abs(3.0 - model.Weights[1, 0]) < 1e-10);
    }

    [Fact]
    public void Predict_ReturnsAccurateEstimates()
    {
        var X = new Matrix(2, 1, [1, 2]);
        var y = new Matrix(2, 1, [10, 20]); // y = 10x + 0
        var model = new LinearRegression();
        model.Fit(X, y);

        var newX = new Matrix(1, 1, [5]);
        var pred = model.Predict(newX);

        Assert.True(Math.Abs(50.0 - pred[0, 0]) < 1e-10);
    }

    [Fact]
    public void Evaluate_ReturnsCorrectMSEAndR2()
    {
        var X = new Matrix(3, 1, [1, 2, 3]);
        var y = new Matrix(3, 1, [2, 4, 6]); // Perfect fit line
        var model = new LinearRegression();
        model.Fit(X, y);

        var (rmse, mse, r2) = model.Evaluate(X, y);

        Assert.True(rmse < 1e-10); // Error should be 0
        Assert.True(mse < 1e-10); // Error should be 0
        Assert.True(Math.Abs(1.0 - r2) < 1e-10); // R2 should be perfect 1.0
    }

    [Fact]
    public void Fit_WithDataFrame_TranslatesCorrectly()
    {
        var featureA = new Column<double>("FeatureA", [1, 2]);
        var target = new Column<double>("Target", [5, 10]);
        var dfX = new DataFrame([featureA]);
        var dfY = new DataFrame([target]);

        var model = new LinearRegression();
        model.Fit(dfX, dfY);

        Assert.True(Math.Abs(5.0 - model.Weights[0, 0]) < 1e-10);
        Assert.True(Math.Abs(0.0 - model.Bias) < 1e-10);
    }
}