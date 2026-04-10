using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Models;

namespace Helix.ML.Tests;

public class DecisionTreeTests
{
    [Fact]
    public void FitPredict_Matrix_ShouldPerfectlySeparate_BinaryData()
    {
        // Arrange: A perfectly clean split. 
        // If Col 0 is < 5.0, it's Class 0. If > 5.0, it's Class 1.
        var xTrain = new Matrix(4, 2, new double[] 
        { 
            1.0, 9.0,   // Class 0
            2.0, 8.0,   // Class 0
            8.0, 1.0,   // Class 1
            9.0, 2.0    // Class 1
        });
        
        var yTrain = new double[] { 0.0, 0.0, 1.0, 1.0 };

        var tree = new DecisionTree(maxDepth: 2);

        // Act
        tree.Fit(xTrain, yTrain);
        
        // Assert: The tree should have found a threshold between 2.0 and 8.0 on Feature 0
        Assert.NotNull(tree.Root);
        Assert.False(tree.Root.IsLeaf);
        Assert.Equal(0, tree.Root.FeatureIndex);
        
        // Predict
        var predictions = tree.Predict(xTrain);
        Assert.Equal(0.0, predictions[0]);
        Assert.Equal(1.0, predictions[3]);
    }

    [Fact]
    public void FitPredict_DataFrame_ShouldHandle_MulticlassStrings()
    {
        // Arrange: A 3-class dataset ("Cat", "Dog", "Bird")
        var weightCol = new Column<double>("Weight", new double[] { 10.0, 12.0, 45.0, 50.0, 1.0, 2.0 });
        var heightCol = new Column<double>("Height", new double[] { 10.0, 11.0, 24.0, 25.0, 4.0, 5.0 });
        
        // Strings mapped to distinct numeric zones
        var targetCol = new Column<string>("Species", new string[] 
        { 
            "Cat", "Cat",   // Medium
            "Dog", "Dog",   // Large
            "Bird", "Bird"  // Small
        });
        
        var df = new DataFrame(new IColumn[] { weightCol, heightCol, targetCol });

        var tree = new DecisionTree(maxDepth: 3);

        // Act
        tree.Fit(df, "Species");

        var predictions = tree.Predict(df);

        // Assert: The auto-encoder should have mapped and predicted the strings perfectly
        Assert.Equal("Cat", predictions[0]);
        Assert.Equal("Dog", predictions[2]);
        Assert.Equal("Bird", predictions[4]);
    }

    [Fact]
    public void Predict_WithoutFitting_ShouldThrowException()
    {
        // Arrange
        var tree = new DecisionTree();
        var xTest = new Matrix(1, 1, new double[] { 5.0 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => tree.Predict(xTest));
    }
}