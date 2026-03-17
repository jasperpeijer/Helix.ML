using Helix.ML.Stat;

namespace Helix.ML.Tests;

public class DataScalerTests
{
    [Fact]
    public void GetStandardized_NormalData_ResultsInZeroMeanAndUnitVariance()
    {
        // Arrange
        double[] data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        // Act
        double[] scaled = DataScaler.GetStandardized(data);

        // Assert
        var (newMean, _, newStdDev) = DescriptiveStats.ComputeSummary(scaled, asSample: false);

        Assert.Equal(0.0, newMean, 5);
        Assert.Equal(1.0, newStdDev, 5);
    }
    
    [Fact]
    public void Standardize_MutatesArrayInPlace()
    {
        // Arrange
        double[] data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        // Act
        DataScaler.Standardize(data); // Modifies the original array directly

        // Assert
        var (newMean, _, newStdDev) = DescriptiveStats.ComputeSummary(data, asSample: false);
        Assert.Equal(0.0, newMean, 5);
        Assert.Equal(1.0, newStdDev, 5);
    }
    
    [Fact]
    public void GetMinMaxScaled_NormalData_BoundsBetweenZeroAndOne()
    {
        // Arrange
        double[] data = [-10.0, 0.0, 10.0, 20.0, 30.0];

        // Act
        double[] scaled = DataScaler.GetMinMaxScaled(data);

        // Assert
        Assert.Equal(0.0, scaled[0], 5); // The minimum (-10) becomes 0
        Assert.Equal(1.0, scaled[4], 5); // The maximum (30) becomes 1
        Assert.Equal(0.5, scaled[2], 5); // The exact middle (10) becomes 0.5
    }
    
    [Fact]
    public void MinMaxScale_MutatesArrayInPlace()
    {
        // Arrange
        double[] data = [10.0, 20.0];

        // Act
        DataScaler.MinMaxScale(data);

        // Assert
        Assert.Equal(0.0, data[0]);
        Assert.Equal(1.0, data[1]);
    }
    
    [Fact]
    public void GetStandardized_FlatlineData_ReturnsAllZeros()
    {
        // Arrange
        double[] data = [5.0, 5.0, 5.0];

        // Act
        double[] scaled = DataScaler.GetStandardized(data);

        // Assert
        Assert.All(scaled, val => Assert.Equal(0.0, val));
    }
}