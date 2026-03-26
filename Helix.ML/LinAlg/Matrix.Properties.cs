using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices.ComTypes;

namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
    #region Matrix Properties

    /// <summary>
    /// Calculates the Trace of the matrix (the sum of the main diagonal elements).
    /// </summary>
    public double Trace(bool allowRectangular = false)
    {
        if (!allowRectangular && !IsSquare)
            throw new InvalidOperationException("The Trace is only defined for square matrices.");

        var sum = 0.0;

        for (var i = 0; i < Math.Min(Rows, Cols); i++) sum += Data[(i * Cols) + i];

        return sum;
    }

    /// <summary>
    /// Creates a smaller matrix by completely removing the specified row and column.
    /// Used primarily for calculating Determinants and Cofactors.
    /// </summary>
    private Matrix GetMinor(int dropRow, int dropCol)
    {
        var result = new Matrix(Rows - 1, Cols - 1);
        var targetRow = 0;

        for (var i = 0; i < Rows; i++)
        {
            if (i == dropRow) continue;

            if (dropCol > 0) Array.Copy(this.Data, i * Cols, result.Data, 
                    targetRow * result.Cols, dropCol);

            if (dropCol < Cols - 1)
            {
                var elementsAfter = Cols - dropCol - 1;
                Array.Copy(this.Data, (i * Cols) + dropCol + 1, result.Data,
                    (targetRow * result.Cols) + dropCol, elementsAfter);
                
            }
            
            targetRow++;
        }
        
        return result;
    }

    /// <summary>
    /// Calculates the determinant of the matrix in O(N^3) time using PLU Decomposition.
    /// </summary>
    public double Determinant()
    {
        if (!IsSquare)
            throw new InvalidOperationException("The Determinant is strictly defined for square matrices.");
        
        // --- THE O(N) FAST PATH ---
        // If the matrix is triangular or diagonal, the determinant is just the product of the main diagonal!
        if (IsUpperTriangular() || IsLowerTriangular())
        {
            var detFast = 1.0;

            for (var i = 0; i < Rows; i++) detFast *= this[i, i];

            return detFast;
        }

        if (Rows % 2 != 0 && IsAntiSymmetric()) return 0.0;
        
        if (Rows == 1) return Data[0];
        
        if (Rows == 2) return (Data[0] * Data[3]) - (Data[1] * Data[2]);

        var (_, _, u, swaps) = PLUDecomposition();
        var determinant = 1.0;

        for (var i = 0; i < Rows; i++)
        {
            determinant *= u[i, i];
        }

        return (swaps % 2 == 0) ? determinant : -determinant;
    }

    /// <summary>
    /// Checks if the matrix is Upper Triangular (all elements below the main diagonal are zero).
    /// </summary>
    public bool IsUpperTriangular(double tolerance = 1e-14)
    {
        for (var i = 1; i < Rows; i++)
        {
            var maxCol = Math.Min(i, Cols);
            
            for (var j = 0; j < maxCol; j++)
            {
                if (Math.Abs(this[i, j]) > tolerance) return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if the matrix is Lower Triangular (all elements above the main diagonal are zero).
    /// </summary>
    public bool IsLowerTriangular(double tolerance = 1e-14)
    {
        for (var i = 0; i < Rows - 1; i++)
        {
            for (var j = i + 1; j < Cols; j++)
            {
                if (Math.Abs(this[i, j]) > tolerance) return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if the matrix is strictly Diagonal (all non-diagonal elements are zero).
    /// </summary>
    public bool IsDiagonal(double tolerance = 1e-14) => IsUpperTriangular(tolerance) && IsLowerTriangular(tolerance);
    
    /// <summary>
    /// Checks if the matrix is perfectly square (Rows == Cols).
    /// </summary>
    public bool IsSquare 
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Rows == Cols;
    }

    /// <summary>
    /// Checks if the matrix is Symmetric (A = A^T).
    /// Uses an optimized loop to avoid memory allocation and allow early exits.
    /// </summary>
    public bool IsSymmetric(double tolerance = 1e-14)
    {
        if (!IsSquare) return false;

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < i; j++)
            {
                if (Math.Abs(this[i, j] - this[j, i]) > tolerance) return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if the matrix is Antisymmetric / Skew-Symmetric (A^T = -A).
    /// </summary>
    public bool IsAntiSymmetric(double tolerance = 1e-14)
    {
        if (!IsSquare) return false;

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j <= i; j++)
            {
                if (Math.Abs(this[i, j] + this[j, i]) > tolerance) return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if a matrix is orthogonal 
    /// </summary>
    public bool IsOrthogonal(double tolerance = 1e-14)
    {
        if (!IsSquare) return false;

        var identity = Identity(Rows);
        var product = this * Transpose();

        return product.IsCloseTo(identity, atol: tolerance);
    }

    /// <summary>
    /// Extracts the elements on the main diagonal into a flat 1D array.
    /// </summary>
    public double[] GetDiagonal()
    {
        var minDim = Math.Min(Rows, Cols);
        var diag = new double[minDim];

        for (var i = 0; i < minDim; i++)
        {
            diag[i] = this[i, i];
        }

        return diag;
    }

    /// <summary>
    /// Calculates the mathematical Norm (magnitude) of the matrix or vector.
    /// </summary>
    public double Norm(NormType normType = NormType.L2)
    {
        return normType switch
        {
            NormType.L1 => NormL1(),
            NormType.L2 => NormL2(),
            _ => throw new NotImplementedException($"Norm type {normType} is not supported.")
        };
    }

    /// <summary>
    /// Calculates the L1 Norm (Manhattan Magnitude) of the matrix/vector.
    /// The sum of the absolute values of all elements.
    /// </summary>
    private double NormL1()
    {
        var data = Data;
        
        if (Data.Length < 100_000) return ProcessChunk(0, data.Length, data);
        
        var globalSum = 0.0;
        var lockObj = new object();

        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, Data.Length), () => 0.0,
            (range, _, localSum) => localSum + ProcessChunk(range.Item1, range.Item2, data),
            (localSum) => { lock (lockObj) globalSum += localSum; }
        );

        return globalSum;
        
        static double ProcessChunk(int start, int end, double[] src)
        {
            var sum = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = start;
            var vOnes = new Vector<double>(1.0);

            for (; i <= end - vectorSize; i += vectorSize)
            {
                var v = new Vector<double>(src, i);
                sum += Vector.Dot(Vector.Abs(v), vOnes);
            }

            for (; i < end; i++) sum += Math.Abs(src[i]);
            
            return sum;
        }
    }

    /// <summary>
    /// Calculates the L2 Norm (Euclidean Magnitude) of the matrix/vector.
    /// Also known mathematically as the Frobenius Norm for 2D matrices.
    /// </summary>
    private double NormL2()
    {
        var data = Data;
        
        if (Data.Length < 100_000) return Math.Sqrt(ProcessChunk(0, data.Length, data));
        
        var globalSum = 0.0;
        var lockObj = new object();

        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, Data.Length), 
            () => 0.0, 
            (range, _, localSum) => localSum + ProcessChunk(range.Item1, range.Item2, data), 
            (localSum) => { lock (lockObj) globalSum += localSum; }
        );
        
        return Math.Sqrt(globalSum);
        
        static double ProcessChunk(int start, int end, double[] src)
        {
            var sumOfSquares = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = start;

            for (; i <= end - vectorSize; i += vectorSize)
            {
                var v = new Vector<double>(src, i);
                sumOfSquares += Vector.Dot(v, v);
            }

            for (; i < end; i++) sumOfSquares += src[i] * src[i];
            
            return sumOfSquares;
        }
    }

    /// <summary>
    /// Calculates the Rank of the matrix (the number of linearly independent rows/columns).
    /// </summary>
    public int Rank(double tolerance = 1e-14) => SVD(tolerance: tolerance).Rank;

    /// <summary>
    /// Calculates the Condition Number (L2-norm) of the matrix.
    /// A high condition number (> 1e4) indicates the matrix is ill-conditioned and sensitive to numerical errors.
    /// </summary>
    public double ConditionNumber(double tolerance = 1e-14) => SVD(tolerance: tolerance).ConditionNumber;
    
    /// <summary>
    /// Checks if the matrix is geometrically a vector (either an N x 1 column or a 1 x N row).
    /// </summary>
    public bool IsVector => Rows == 1 || Cols == 1;

    /// <summary>
    /// Calculates the scalar Quadratic Form (x^T * A * x) for a given vector x.
    /// Executed in highly optimized O(N^2) time without allocating any intermediate matrices.
    /// </summary>
    public double QuadraticForm(Matrix x)
    {
        if (!IsSquare) 
            throw new InvalidOperationException("Matrix must be square to calculate the Quadratic Form.");
        if (!x.IsVector) 
            throw new ArgumentException("The input x must be a vector.");
        if (x.Data.Length != Rows) 
            throw new ArgumentException("The vector length must match the matrix dimensions.");

        double result = 0;

        for (var i = 0; i < Rows; i++)
        {
            double rowSum = 0;

            for (var j = 0; j < Cols; j++)
            {
                rowSum += this[i, j] * x.Data[j];
            }
            
            result += rowSum * x.Data[i];
        }
        
        return result;
    }
    
    /// <summary>
    /// Checks if the matrix forms a perfect, upward-facing "bowl" geometry.
    /// Evaluates in O(N^3 / 3) time by utilizing the Cholesky fast-path.
    /// </summary>
    public bool IsPositiveDefinite(double tolerance = 1e-14)
    {
        return TryCholeskyDecomposition(out _, tolerance);
    }

    /// <summary>
    /// Evaluates the geometric shape of the matrix's quadratic form.
    /// Uses high-speed Cholesky checks for strict definiteness before falling back to Eigenvalue analysis.
    /// </summary>
    public Definiteness Definiteness(double tolerance = 1e-14)
    {
        if (!IsSquare || !IsSymmetric(tolerance)) return LinAlg.Definiteness.Unknown;

        if (IsPositiveDefinite(tolerance)) return LinAlg.Definiteness.PositiveDefinite;
        
        if ((-this).IsPositiveDefinite(tolerance)) return LinAlg.Definiteness.NegativeDefinite;

        var eigenvalues = Eigenvalues();

        var hasPositive = false;
        var hasNegative = false;
        var hasZero = false;

        foreach (var val in Data)
        {
            if (val > tolerance) hasPositive = true;
            else if (val < -tolerance) hasNegative = true;
            else hasZero = true;
        }

        if ( hasPositive && !hasNegative && hasZero) return LinAlg.Definiteness.PositiveSemidefinite;
        if (!hasPositive &&  hasNegative && hasZero) return LinAlg.Definiteness.NegativeSemidefinite;
        if ( hasPositive &&  hasNegative)            return LinAlg.Definiteness.Indefinite;

        return LinAlg.Definiteness.Unknown;
    }

    #endregion
}

public enum Definiteness
{
    PositiveDefinite,
    PositiveSemidefinite,
    NegativeDefinite,
    NegativeSemidefinite,
    Indefinite,
    Unknown
}