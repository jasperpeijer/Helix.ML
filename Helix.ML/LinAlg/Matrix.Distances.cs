namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
    /// <summary>
    /// Calculates the Euclidean Distance (straight-line physical distance) between two matrices/vectors.
    /// </summary>
    public double EuclideanDistance(Matrix other) => (this - other).NormL2();

    /// <summary>
    /// Calculates the Manhattan Distance (grid-like block distance) between two matrices/vectors.
    /// </summary>
    public double ManhattanDistance(Matrix other) => (this - other).NormL1();
    
    /// <summary>
    /// Calculates the Cosine Similarity between two matrices/vectors (-1.0 to 1.0).
    /// </summary>
    public double CosineSimilarity(Matrix other)
    {
        var dotProduct = DotProduct(other);
        double normA = this.NormL2(), normB = other.NormL2();

        if (normA == 0.0 || normB == 0.0) return 0.0;
        
        return dotProduct / (normA * normB);
    }
}