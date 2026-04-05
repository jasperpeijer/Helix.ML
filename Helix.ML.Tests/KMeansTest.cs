using Helix.ML.LinAlg;
using Helix.ML.Models;

namespace Helix.ML.Tests;

public class KMeansTest
{
    [Fact]
    public void FitPredict_ShouldCorrectlyGroupObviouslySeparatedData()
    {
        // Arrange: Create 4 points. 
        // Rows 0 and 1 are near (1,1). Rows 2 and 3 are near (100, 100).
        var x = new Matrix(4, 2, new double[]
        {
            1.0, 1.0,   // Row 0: Group A
            1.1, 1.1,   // Row 1: Group A
            100.0, 100.0, // Row 2: Group B
            100.1, 100.1  // Row 3: Group B
        });

        var kmeans = new KMeans(clusters: 2);

        // Act
        var clusters = kmeans.FitPredict(x);

        // Assert: We don't care if Group A is called "0" or "1". 
        // We only care that the math grouped them together!
        Assert.Equal(4, clusters.Length); // Must return 4 predictions
        
        Assert.Equal(clusters[0], clusters[1]); // Row 0 and 1 must match
        Assert.Equal(clusters[2], clusters[3]); // Row 2 and 3 must match
        
        Assert.NotEqual(clusters[0], clusters[2]); // Group A must not equal Group B
    }
    
    [Fact]
    public void Predict_ShouldMatchFitPredict_ForSameData()
    {
        // Arrange
        var x = new Matrix(4, 2, new double[]
        {
            1.0, 1.0, 1.1, 1.1, 100.0, 100.0, 100.1, 100.1
        });
        
        var kmeans = new KMeans(clusters: 2);

        // Act
        var fitClusters = kmeans.FitPredict(x);
        var predictClusters = kmeans.Predict(x);

        // Assert: Once trained, passing the same data through Predict 
        // should yield the exact same geometric assignments.
        for (int i = 0; i < x.Rows; i++)
        {
            Assert.Equal(fitClusters[i], predictClusters[i]);
        }
    }
    
    [Fact]
    public void FitPredict_ShouldThrowException_WhenClustersExceedRows()
    {
        // Arrange: 3 rows of data
        var x = new Matrix(3, 2, new double[] { 1, 1, 2, 2, 3, 3 });
        
        // Act & Assert: Asking for 4 clusters should mathematically crash, 
        // so we verify our safety check works!
        var kmeans = new KMeans(clusters: 4);
        
        Assert.Throws<ArgumentException>(() => kmeans.FitPredict(x));
    }
    
    [Fact]
    public void FitPredict_AllLabelsShouldBeWithinBounds()
    {
        // Arrange: Random scattered data
        var x = new Matrix(10, 2, new double[] 
        { 
            1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16, 17,18, 19,20 
        });
        
        int requestedClusters = 3;
        var kmeans = new KMeans(clusters: requestedClusters);

        // Act
        var clusters = kmeans.FitPredict(x);

        // Assert: No label should be less than 0 or greater than 2
        for (int i = 0; i < clusters.Length; i++)
        {
            int label = int.Parse(clusters[i]); // Because we output a Column<string>
            Assert.True(label >= 0 && label < requestedClusters);
        }
    }
}