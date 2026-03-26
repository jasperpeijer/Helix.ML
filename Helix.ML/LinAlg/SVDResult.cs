namespace Helix.ML.LinAlg;

/// <summary>
/// A dedicated container for the results of a Singular Value Decomposition.
/// Caches the U, S, and V matrices so properties like Rank and PseudoInverse 
/// can be computed instantly without recalculating the SVD.
/// </summary>
// ReSharper disable once InconsistentNaming
public class SVDResult
{
    public Matrix U { get; }
    public Matrix S { get; }
    public Matrix V { get; }
    public int Rank => GetRank(_tolerance);
    public double ConditionNumber => GetConditionNumber(_tolerance);

    private readonly double _tolerance;

    internal SVDResult(Matrix u, Matrix s, Matrix v, double tolerance = 1e-14)
    {
        U = u;
        S = s;
        V = v;
        _tolerance = tolerance;
    }
    
    /// <summary>
    /// Gets the Rank of the original matrix (O(N) operation on the cached S matrix).
    /// </summary>
    private int GetRank(double tolerance = 1e-14)
    {
        var rank = 0;
        var minDim = Math.Min(S.Rows, S.Cols);
        
        for (var i = 0; i < minDim; i++)
        {
            if (Math.Abs(S[i, i]) > tolerance) rank++;
        }
        return rank;
    }

    /// <summary>
    /// Gets the Condition Number of the original matrix (O(1) operation).
    /// </summary>
    private double GetConditionNumber(double tolerance = 1e-14)
    {
        var minDim = Math.Min(S.Rows, S.Cols);
        var maxS = S[0, 0]; 
        var minS = S[minDim - 1, minDim - 1]; 

        if (Math.Abs(minS) < tolerance) return double.PositiveInfinity; 
        
        return maxS / minS;
    }

    /// <summary>
    /// Calculates the Moore-Penrose Pseudoinverse using Singular Value Decomposition (SVD).
    /// Automatically and safely handles square, tall, wide, and rank-deficient matrices.
    /// </summary>
    public Matrix PseudoInverse(double tolerance = 1e-14)
    {
        var sPlus = Matrix.Zeros(S.Cols, S.Rows);
        var minDim = Math.Min(S.Rows, S.Cols);

        for (var i = 0; i < minDim; i++)
        {
            if (S[i, i] > tolerance)
            {
                sPlus[i, i] = 1.0 / S[i, i];
            }
        }

        return V * sPlus * U.T;
    }
}