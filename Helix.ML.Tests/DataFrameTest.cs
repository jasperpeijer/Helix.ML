using Helix.ML.Data;
using Helix.ML.LinAlg;

namespace Helix.ML.Tests;

public class DataFrameTest
{
    [Fact]
    public void Constructor_NoIndicesProvided_DefaultsToNumericStrings()
    {
        var colA = new Column<double>("A", [1, 3]);
        var colB = new Column<double>("B", [2, 4]);
        
        var df = new DataFrame([colA, colB]);

        Assert.Equal("0", df.Indices[0]);
        Assert.Equal("1", df.Indices[1]);
    }

    [Fact]
    public void Constructor_CustomIndicesProvided_AppliesCorrectly()
    {
        var colA = new Column<double>("A", [1, 3]);
        var colB = new Column<double>("B", [2, 4]);
        
        var df = new DataFrame([colA, colB], ["Row1", "Row2"]);

        Assert.Equal("Row1", df.Indices[0]);
        Assert.Equal("Row2", df.Indices[1]);
    }

    [Fact]
    public void Indexer_ValidColumn_ReturnsCorrectColumn()
    {
        var col1 = new Column<double>("Feature1", [10, 30, 50]);
        var col2 = new Column<double>("Feature2", [20, 40, 60]);
        var df = new DataFrame([col1, col2]);

        // Extract the column and cast it back to its strong type
        var feature2 = (Column<double>)df["Feature2"];

        Assert.Equal(3, feature2.Length);
        Assert.Equal(20.0, feature2[0]);
        Assert.Equal(60.0, feature2[2]);
    }

    [Fact]
    public void Select_ValidColumns_ReturnsSlicedDataFrameWithIndices()
    {
        var col1 = new Column<double>("Col1", [1, 4]);
        var col2 = new Column<double>("Col2", [2, 5]);
        var col3 = new Column<double>("Col3", [3, 6]);
        var df = new DataFrame([col1, col2, col3], ["R1", "R2"]);

        var slicedDf = df.Select("Col3", "Col1"); 

        Assert.Equal(2, slicedDf.Cols);
        Assert.Equal("Col3", slicedDf.ColumnNames[0]);
        Assert.Equal("R2", slicedDf.Indices[1]); 
        
        // Assert the data survived
        var c3 = (Column<double>)slicedDf["Col3"];
        Assert.Equal(3.0, c3[0]); 
    }

    [Fact]
    public void Describe_CalculatesStatistics_IgnoresStrings()
    {
        var colA = new Column<double>("FeatureA", [2, 4, 6]);
        var colB = new Column<string>("FeatureB", ["X", "Y", "Z"]); // String column!
        var colC = new Column<double>("FeatureC", [10, 20, 30]);
        var df = new DataFrame([colA, colB, colC]);

        var stats = df.Describe();

        Assert.Equal(5, stats.Rows);
        Assert.Equal(2, stats.Cols); // Notice FeatureB is gone!
        
        Assert.Equal("Count", stats.Indices[0]);
        Assert.Equal("Max", stats.Indices[4]);

        var statA = (Column<double>)stats["FeatureA"];
        var statC = (Column<double>)stats["FeatureC"];

        // Assert Math (Mean)
        Assert.Equal(4.0, statA[1]); 
        Assert.Equal(20.0, statC[1]); 

        // Assert Math (Max)
        Assert.Equal(6.0, statA[4]); 
        Assert.Equal(30.0, statC[4]); 
    }
}