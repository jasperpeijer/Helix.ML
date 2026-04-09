using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Models;
using Helix.ML.Stat;

namespace Helix.ML.Tests;

public class LogisticRegressionTests
{
    [Fact]
    public void FitPredict_ShouldCorrectlyClassify_ObviouslySeparatedData()
    {
        // Arrange: Group 0 is near the origin, Group 1 is far away.
        var xTrain = new Matrix(4, 2, new double[] 
        { 
            0.1, 0.1,    // Row 0 -> Class 0
            0.2, 0.2,    // Row 1 -> Class 0
            10.0, 10.0,  // Row 2 -> Class 1
            11.0, 11.0   // Row 3 -> Class 1
        });
        
        var yTrain = new double[] { 0.0, 0.0, 1.0, 1.0 };

        // Act: Use a decent learning rate so it can walk down the hill
        var logReg = new LogisticRegression(learningRate: 0.1, maxIterations: 1000);
        logReg.Fit(xTrain, yTrain);

        var xTest = new Matrix(2, 2, new double[] 
        { 
            0.15, 0.15,  // Should be close to 0.0
            10.5, 10.5   // Should be close to 1.0
        });

        var probabilities = logReg.PredictProbabilities(xTest);

        // Assert: The math should confidently push the probabilities toward the extremes
        Assert.True(probabilities[0] < 0.5, $"Expected probability < 0.5, but got {probabilities[0]}");
        Assert.True(probabilities[1] > 0.5, $"Expected probability > 0.5, but got {probabilities[1]}");
    }

    [Fact]
    public void Fit_ShouldSupport_OneDimensionalMatrixTargets()
    {
        // Arrange: Test your Array.Copy optimization for 1xN matrices
        var xTrain = new Matrix(3, 1, new double[] { 1, 2, 3 });
        
        // A 1x3 Matrix instead of a flat double[]
        var yTrainMatrix = new Matrix(1, 3, new double[] { 0.0, 0.0, 1.0 });

        var logReg = new LogisticRegression(learningRate: 0.01, maxIterations: 10);
        
        // Act & Assert: This should not throw any exceptions
        var exception = Record.Exception(() => logReg.Fit(xTrain, yTrainMatrix));
        Assert.Null(exception);
        Assert.NotNull(logReg.Weights);
    }

    [Fact]
    public void PredictProbabilities_ShouldStrictlyOutput_BetweenZeroAndOne()
    {
        // Arrange
        var xTrain = new Matrix(3, 1, new double[] { 1, 5, 10 });
        var yTrain = new double[] { 0.0, 0.0, 1.0 };
        
        var logReg = new LogisticRegression(learningRate: 0.01, maxIterations: 100);
        logReg.Fit(xTrain, yTrain);

        // We test an extreme outlier to ensure the Sigmoid function squashes it properly
        var xTest = new Matrix(1, 1, new double[] { 9999.0 });

        // Act
        var probabilities = logReg.PredictProbabilities(xTest);

        // Assert
        Assert.True(probabilities[0] >= 0.0 && probabilities[0] <= 1.0, 
            "Sigmoid function failed to compress probability bounds.");
    }

    [Fact]
    public void DataFramePipeline_ShouldMapStringsAndScale_Correctly()
    {
        // Arrange: A mini-dataset using raw strings and drastically different scales
        var xCol1 = new Column<double>("Age", new double[] { 22, 25, 55, 60 });
        var xCol2 = new Column<double>("Income", new double[] { 35000, 42000, 160000, 175000 });
        var yCol = new Column<string>("Premium", new string[] { "No", "No", "Yes", "Yes" });
        
        var df = new DataFrame(new IColumn[] { xCol1, xCol2, yCol });

        var logReg = new LogisticRegression(learningRate: 0.1, maxIterations: 1000);
        
        // Act: Use the pipeline with Standardization
        logReg.Fit(df, "Premium", "Yes", ScalerType.Standardize);

        // Create a new unseen test customer (Older, High Income)
        var testAge = new Column<double>("Age", new double[] { 58 });
        var testIncome = new Column<double>("Income", new double[] { 170000 });
        var testDf = new DataFrame([testAge, testIncome]);

        var predictions = logReg.Predict(testDf, threshold: 0.5);

        // Assert: The pipeline should have scaled the new data and translated the prediction back to "Yes"
        Assert.Equal("Yes", predictions[0]);
    }
}