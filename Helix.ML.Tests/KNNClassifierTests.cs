using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Models;

namespace Helix.ML.Tests;

public class KNNClassifierTests
{
    [Fact]
    public void Predict_ShouldCorrectlyClassify_ObviouslySeparatedData()
    {
        // Arrange: Group A is near (0,0), Group B is near (100,100)
        var xTrain = new Matrix(4, 2, new double[] 
        { 
            0.0, 0.0,    // Row 0 -> Group A
            1.0, 1.0,    // Row 1 -> Group A
            100.0, 100.0, // Row 2 -> Group B
            101.0, 101.0  // Row 3 -> Group B
        });
        
        var yTrain = new Column<string>("Labels", ["A", "A", "B", "B"]);

        var knn = new KNNClassifier(k: 1);
        knn.Fit(xTrain, yTrain);

        // Act: Test points clearly belonging to each group
        var xTest = new Matrix(2, 2, new double[] 
        { 
            0.5, 0.5,    // Should be A
            100.5, 100.5 // Should be B
        });
        
        var predictions = knn.Predict(xTest);

        // Assert
        Assert.Equal("A", predictions[0]);
        Assert.Equal("B", predictions[1]);
    }

    [Fact]
    public void Predict_ShouldUseMajorityVote_WhenNeighborsConflict()
    {
        // Arrange: A chaotic boundary where neighbors are mixed
        var xTrain = new Matrix(5, 2, new double[] 
        { 
            1, 1, // A
            2, 2, // A
            3, 3, // B (Outlier in A's territory)
            8, 8, // B
            9, 9  // B
        });
        
        var yTrain = new Column<string>("Labels", ["A", "A", "B", "B", "B"]);

        // We use K=3. The closest points to (2,2) are (1,1)[A], (2,2)[A], and (3,3)[B].
        // The vote should be 2 for A, 1 for B.
        var knn = new KNNClassifier(k: 3);
        knn.Fit(xTrain, yTrain);

        var xTest = new Matrix(1, 2, new double[] { 2.1, 2.1 });
        
        // Act
        var predictions = knn.Predict(xTest);

        // Assert: "A" should win the 2-to-1 vote
        Assert.Equal("A", predictions[0]);
    }

    [Fact]
    public void Fit_ShouldThrowException_IfRowCountsMismatch()
    {
        var xTrain = new Matrix(3, 2, new double[] { 1, 1, 2, 2, 3, 3 });
        var yTrain = new Column<string>("Labels", ["A", "B"]); // Only 2 labels!

        var knn = new KNNClassifier(k: 1);

        Assert.Throws<ArgumentException>(() => knn.Fit(xTrain, yTrain));
    }
}