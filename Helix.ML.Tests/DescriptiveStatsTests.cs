using Helix.ML.Stat;

namespace Helix.ML.Tests;

public class DescriptiveStatsTests
{
    private readonly double[] _testData = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    [Fact]
    public void ComputeSummary_Population_ReturnsCorrectStatistics()
    {
        // Act
        var result = DescriptiveStats.ComputeSummary(_testData, asSample: false);
        
        // Assert
        Assert.Equal(5.0, result.Mean, 5);
        Assert.Equal(4.0, result.Variance, 5);
        Assert.Equal(2.0, result.StdDev, 5);
    }

    [Fact]
    public void ComputeSummary_Sample_ReturnsCorrectStatistics()
    {
        // Arrange
        double expectedSampleVariance = 32.0 / 7;
        double expectedSampleStdDev = Math.Sqrt(expectedSampleVariance);
            
        // Act
        var result = DescriptiveStats.ComputeSummary(_testData, asSample: true);

        // Assert
        Assert.Equal(5.0, result.Mean, 5);
        Assert.Equal(expectedSampleVariance, result.Variance, 5);
        Assert.Equal(expectedSampleStdDev, result.StdDev, 5);
    }

    [Fact]
    public void Mean_StandaloneMethod_ReturnsCorrectAverage()
    {
        // Act
        double result = DescriptiveStats.Mean(_testData);
        
        // Assert
        Assert.Equal(5.0, result, 5);
    }

    [Fact]
    public void ComputeSummary_EmptyArray_ReturnsZeros()
    {
        // Act
        var result = DescriptiveStats.ComputeSummary([]);
        
        // Assert
        Assert.Equal(0.0, result.Mean);
        Assert.Equal(0.0, result.Variance);
        Assert.Equal(0.0, result.StdDev);
    }

    [Fact]
    public void ComputeSummary_OneElementArray_ReturnsZerosForVarianceAndStdDev()
    {
        // Act
        var result = DescriptiveStats.ComputeSummary([2.0]);
        
        // Assert
        Assert.Equal(2.0, result.Mean);
        Assert.Equal(0.0, result.Variance);
        Assert.Equal(0.0, result.StdDev);
    }

    [Fact]
    public void ComputeSummary_HighPrecision_AvoidsCatastrophicCancellation()
    {
        // Arrange
        // We use a massive offset. Squaring these numbers in the naïve formula 
        // would exceed the 15-digit precision limit of a standard double.
        double offset = 1000000000.0;
        double[] data = [offset + 1.0, offset + 2.0, offset + 3.0];
        
        // Act
        var result = DescriptiveStats.ComputeSummary(data, asSample: false);
        
        // Assert
        Assert.Equal(offset + 2.0, result.Mean, 10);
        
        // The population variance of [1, 2, 3] is exactly 2/3 (0.666666...).
        // If precision was lost, this would evaluate to 0 or a wild fraction.
        Assert.Equal(2.0 / 3.0, result.Variance, 10);
    }

    [Fact]
    public void ComputeSummary_MicroscopicVariance_AvoidsUnderFlow()
    {
        // Arrange
        // The numbers are around 1.0, but the differences are microscopic (1e-7).
        double baseVal = 1.0;
        double tiny = 1e-7;
        double[] data = { baseVal - tiny, baseVal, baseVal + tiny };
        
        // The exact population variance of [-tiny, 0, +tiny] is:
        // (tiny^2 + 0 + tiny^2) / 3 = (2/3) * tiny^2
        double expectedVariance = (2.0 / 3.0) * (tiny * tiny);
        
        // Act
        var result = DescriptiveStats.ComputeSummary(data, asSample: false);
        
        // Assert
        Assert.Equal(baseVal, result.Mean, 10);
        
        // Note: xUnit's standard Assert.Equal(expected, actual, precision) relies on decimal places, 
        // which breaks down when checking 15 decimal places deep. 
        // Instead, we assert that the mathematical difference is effectively zero.
        double difference = Math.Abs(expectedVariance - result.Variance);
        Assert.True(difference < 1e-15, $"Precision lost! Expected: {expectedVariance}, Actual: {result.Variance}");
    }
}