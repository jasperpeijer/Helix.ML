using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
    #region Solvers
    
    /// <summary>
    /// Calculates the Inverse analytically, utilizing a pre-computed determinant to save CPU cycles.
    /// WARNING: Passing an incorrect determinant will result in a mathematically invalid inverse.
    /// </summary>
    public Matrix Inverse(double knownDeterminant, double tolerance = 1e-14)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Only square matrices can be inverted.");
        
        // --- THE O(1) FAST PATH ---
        // If the matrix is orthogonal, the inverse is literally just the transpose!
        if (IsOrthogonal(tolerance)) return this.T;

        if (Math.Abs(knownDeterminant) < tolerance)
            throw new InvalidOperationException("Matrix is singular (determinant is 0) and cannot be inverted.");
        
        if (Rows == 1) return new Matrix(1, 1, [1.0 / Data[0]]);

        if (Rows == 2)
        {
            var invDet = 1.0 / knownDeterminant;
            
            return new Matrix(2, 2, [
                Data[3] * invDet, -Data[1] * invDet,
                -Data[2] * invDet, Data[0] * invDet
            ]);
        }

        var cofactors = new Matrix(Rows, Cols);

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < Cols; j++)
            {
                var minor = GetMinor(i, j);
                var minorDeterminant = minor.Determinant();

                var sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                cofactors[i, j] = minorDeterminant * sign;
            }
        }

        return cofactors.T / knownDeterminant;
    }
    
    /// <summary>
    /// Calculates the Inverse of the matrix analytically using the Adjugate/Cofactor method.
    /// Computes the determinant automatically.
    /// </summary>
    public Matrix Inverse(double tolerance = 1e-14) => Inverse(Determinant(), tolerance);

    /// <summary>
    /// Syntactic sugar for Matrix Inverse (!A).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator !(Matrix matrix) => matrix.Inverse();

    /// <summary>
    /// Calculates the Moore-Penrose Pseudoinverse.
    /// Automatically handles Square, Tall (Overdetermined), and Wide (Underdetermined) matrices.
    /// </summary>
    /// TODO: Implement using SVD
    public Matrix PseudoInverse(double tolerance = 1e-14)
    {
        if (IsSquare) return this.Inverse();

        try
        {
            // Left Inverse (Tall Matrix)
            if (Rows > Cols)
            {
                var aTa = this.T * this;
                return aTa.Inverse(tolerance) * this.T;
            }
            
            // Right Inverse (Wide Matrix)
            var aaT = this * this.T;
            return this.T * aaT.Inverse(tolerance);
        }
        catch (InvalidOperationException ex)
        {
            // THE ARCHITECTURAL FAIL-SAFE
            // If the inner inverse throws a singular matrix exception, it means the data 
            // has linearly dependent columns (multicollinearity).
            throw new InvalidOperationException(
                "Algebraic Pseudoinverse failed due to rank deficiency. " +
                "TODO: Implement Singular Value Decomposition (SVD) for robust Moore-Penrose computation.", ex);
        }

    }

    /// <summary>
    /// Performs LU Decomposition (Doolittle Algorithm) to factor a square matrix into 
    /// a Lower triangular matrix (L) and an Upper triangular matrix (U), such that A = L * U.
    /// </summary>
    /// <returns>A tuple containing (L, U).</returns>
    // ReSharper disable once InconsistentNaming
    public (Matrix L, Matrix U) LUDecomposition()
    {
        // Not fully sure how this algorithm works, but it's a computed version
        // of Doolittle's method for LU decomposition.
        if (!IsSquare) 
            throw new InvalidOperationException("LU Decomposition is only defined for square matrices.");

        var n = Rows;
        var l = Identity(n);
        var u = Zeros(n, n);

        for (var i = 0; i < n; i++)
        {
            for (var k = i; k < n; k++)
            {
                var sum = 0.0;

                for (var j = 0; j < i; j++)
                {
                    sum += l[i, j] * u[j, k];
                }
                
                u[i, k] = this[i, k] - sum;
            }

            for (var k = i + 1; k < n; k++)
            {
                var sum = 0.0;

                for (var j = 0; j < i; j++)
                {
                    sum += l[k, j] * u[j, i];
                }

                if (u[i, i] == 0.0)
                {
                    throw new InvalidOperationException(
                        "Matrix cannot be LU decomposed without row swaps. " +
                        "A zero pivot was encountered. TODO: Implement PLU (Partial Pivoting).");
                }
                
                l[k, i] = (this[k, i] - sum) / u[i, i];
            }
        }

        return (l, u);
    }
    
    #endregion Solvers
}