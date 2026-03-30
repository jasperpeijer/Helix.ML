using Helix.ML.Data;
using Helix.ML.LinAlg;

namespace Helix.ML.Tests;

public class DataFrameTest
{
    [Fact]
    public void Constructor_ValidInput_CreatesDataFrame()
    {
        // Arrange
        var m = new Matrix(2, 3, [
            1, 2, 3,
            4, 5, 6
        ]);
        var cols = new List<string> { "A", "B", "C" };

        // Act
        var df = new DataFrame(m, cols);

        // Assert
        Assert.Equal(2, df.Rows);
        Assert.Equal(3, df.Cols);
        Assert.Equal("A", df.Columns[0]);
        Assert.Equal("C", df.Columns[2]);
    }

    [Fact]
    public void Constructor_DimensionMismatch_ThrowsArgumentException()
    {
        var m = new Matrix(2, 3); // 3 columns
        var cols = new List<string> { "A", "B" }; // Only 2 names

        Assert.Throws<ArgumentException>(() => new DataFrame(m, cols));
    }

    [Fact]
    public void Indexer_ValidColumn_ReturnsCorrectMatrix()
    {
        // Arrange
        var m = new Matrix(3, 2, [
            10, 20,
            30, 40,
            50, 60
        ]);
        var df = new DataFrame(m, ["Feature1", "Feature2"]);

        // Act
        Matrix feature2 = df["Feature2"];

        // Assert
        Assert.Equal(3, feature2.Rows);
        Assert.Equal(1, feature2.Cols);
        Assert.Equal(20.0, feature2[0, 0]);
        Assert.Equal(40.0, feature2[1, 0]);
        Assert.Equal(60.0, feature2[2, 0]);
    }

    [Fact]
    public void Indexer_InvalidColumn_ThrowsKeyNotFoundException()
    {
        var m = new Matrix(2, 2);
        var df = new DataFrame(m, ["A", "B"]);

        Assert.Throws<KeyNotFoundException>(() => df["GhostColumn"]);
    }

    [Fact]
    public void Select_ValidColumns_ReturnsSlicedDataFrame()
    {
        // Arrange
        var m = new Matrix(2, 4, [
            1, 2, 3, 4,
            5, 6, 7, 8
        ]);
        var df = new DataFrame(m, ["Col1", "Col2", "Col3", "Col4"]);

        // Act
        var slicedDf = df.Select("Col4", "Col2"); // Swap order to test mapping

        // Assert
        Assert.Equal(2, slicedDf.Rows);
        Assert.Equal(2, slicedDf.Cols);
        
        Assert.Equal("Col4", slicedDf.Columns[0]);
        Assert.Equal("Col2", slicedDf.Columns[1]);

        // Row 0
        Assert.Equal(4.0, slicedDf.CoreMatrix[0, 0]);
        Assert.Equal(2.0, slicedDf.CoreMatrix[0, 1]);
    }

    [Fact]
    public void Select_InvalidColumn_ThrowsKeyNotFoundException()
    {
        var m = new Matrix(2, 2);
        var df = new DataFrame(m, ["A", "B"]);

        Assert.Throws<KeyNotFoundException>(() => df.Select("A", "GhostColumn"));
    }
}