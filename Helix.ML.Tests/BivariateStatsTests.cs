using Helix.ML.Stat;

namespace Helix.ML.Tests;

public class BivariateStatsTests
{
    [Fact]
    public void Compute_PerfectPositiveCorrelation_ReturnsOne()
    {
        // Arrange: Y is exactly 2 times X
        double[] x = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] y = [2.0, 4.0, 6.0, 8.0, 10.0];

        // Act
        var result = BivariateStats.ComputeBivariateStats(x, y);

        // Assert
        Assert.Equal(1.0, result.PearsonCorrelation, 5);
        Assert.True(result.Covariance > 0, "Covariance should be positive");
    }
    
    [Fact]
    public void Compute_PerfectNegativeCorrelation_ReturnsNegativeOne()
    {
        // Arrange: Y moves exactly opposite to X
        double[] x = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] y = [-1.0, -2.0, -3.0, -4.0, -5.0];

        // Act
        var result = BivariateStats.ComputeBivariateStats(x, y);

        // Assert
        Assert.Equal(-1.0, result.PearsonCorrelation, 5);
        Assert.True(result.Covariance < 0, "Covariance should be negative");
    }

    [Fact]
    public void Compute_ZeroVariance_ReturnsZeroCorrelation()
    {
        // Arrange: X never changes, so there is no relationship to measure
        double[] x = [5.0, 5.0, 5.0, 5.0];
        double[] y = [1.0, 2.0, 3.0, 4.0];

        // Act
        var result = BivariateStats.ComputeBivariateStats(x, y);

        // Assert
        Assert.Equal(0.0, result.PearsonCorrelation, 5);
        Assert.Equal(0.0, result.Covariance, 5);
    }

    [Fact]
    public void Compute_MismatchedLengths_ThrowsArgumentException()
    {
        // Arrange
        double[] x = [1.0, 2.0, 3.0];
        double[] y = [1.0, 2.0]; // Missing a data point

        // Act & Assert
        Assert.Throws<ArgumentException>(() => BivariateStats.ComputeBivariateStats(x, y));
    }
    
    [Fact]
    public void SpearmanCorrelation_NonLinearMonotonicData_ReturnsPerfectOne()
    {
        // Arrange: A perfect exponential curve (y = x^3)
        double[] x = [1.0, 2.0, 3.0, 4.0, 5.0];
        double[] y = [1.0, 8.0, 27.0, 64.0, 125.0];

        // Act
        var pearson = BivariateStats.ComputeBivariateStats(x, y).PearsonCorrelation;
        var spearman = BivariateStats.SpearmanCorrelation(x, y);

        // Assert
        // Pearson will be high, but NOT perfect (usually around 0.95 to 0.98) because it curves
        Assert.True(pearson is < 1.0 and > 0.9);
        
        // Spearman only looks at the ranks. Since Y always goes up when X goes up, it is exactly 1.0
        Assert.Equal(1.0, spearman, 5);
    }
}