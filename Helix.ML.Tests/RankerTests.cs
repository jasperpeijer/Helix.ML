using Helix.ML.Stat;

namespace Helix.ML.Tests;

public class RankerTests
{
    [Fact]
    public void GetRanks_NoTies_ReturnsSequentialRanks()
    {
        // Arrange
        double[] data = [50.0, 10.0, 100.0];

        // Act
        double[] ranks = Ranker.GetRanks(data);

        // Assert
        // 10.0 is the smallest (Rank 1), 50.0 is middle (Rank 2), 100.0 is largest (Rank 3)
        Assert.Equal(2.0, ranks[0]); 
        Assert.Equal(1.0, ranks[1]); 
        Assert.Equal(3.0, ranks[2]); 
    }
    
    [Fact]
    public void GetRanks_WithTies_AveragesTheTiedRanks()
    {
        // Arrange
        double[] data = [10.5, 20.1, 20.1, 99.9];

        // Act
        double[] ranks = Ranker.GetRanks(data);

        // Assert
        Assert.Equal(1.0, ranks[0]); // 10.5 is Rank 1
        Assert.Equal(2.5, ranks[1]); // 20.1 is tied for Ranks 2 and 3. Average is 2.5
        Assert.Equal(2.5, ranks[2]); // 20.1 is tied for Ranks 2 and 3. Average is 2.5
        Assert.Equal(4.0, ranks[3]); // 99.9 is Rank 4
    }
}