using System.Numerics;
using System.Runtime.CompilerServices;

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
        {
            throw new InvalidOperationException("The Trace is only defined for square matrices.");
        }

        double sum = 0.0;

        for (int i = 0; i < Math.Min(Rows, Cols); i++)
        {
            sum += Data[(i * Cols) + i];
        }

        return sum;
    }

    /// <summary>
    /// Creates a smaller matrix by completely removing the specified row and column.
    /// Used primarily for calculating Determinants and Cofactors.
    /// </summary>
    private Matrix GetMinor(int dropRow, int dropCol)
    {
        var result = new Matrix(Rows - 1, Cols - 1);
        int targetRow = 0;

        for (var i = 0; i < Rows; i++)
        {
            if (i == dropRow) continue;

            if (dropCol > 0)
            {
                Array.Copy(this.Data, i * Cols, result.Data, 
                    targetRow * result.Cols, dropCol);
            }

            if (dropCol < Cols - 1)
            {
                int elementsAfter = Cols - dropCol - 1;
                Array.Copy(this.Data, (i * Cols) + dropCol + 1, result.Data,
                    (targetRow * result.Cols) + dropCol, elementsAfter);
                
            }
            
            targetRow++;
        }
        
        return result;
    }

    /// <summary>
    /// Calculates the Determinant of a square matrix using Laplace Expansion,
    /// with an O(N) fast-path for triangular matrices.
    /// Warning: This runs in O(N!) time. Do not use on massive matrices.
    /// </summary>
    public double Determinant()
    {
        if (!IsSquare)
        {
            throw new InvalidOperationException("The Determinant is strictly defined for square matrices.");
        }
        
        // --- THE O(N) FAST PATH ---
        // If the matrix is triangular or diagonal, the determinant is just the product of the main diagonal!
        if (IsUpperTriangular() || IsLowerTriangular())
        {
            double det = 1.0;

            for (int i = 0; i < Rows; i++)
            {
                det *= this[i, i];
            }

            return det;
        }

        if (Rows % 2 != 0 && IsAntiSymmetric())
        {
            return 0.0;
        }
        
        if (Rows == 1) return Data[0];
        
        if (Rows == 2) return (Data[0] * Data[3]) - (Data[1] * Data[2]);

        double determinant = 0.0;
        int sign = 1;

        for (int j = 0; j < Cols; j++)
        {
            double element = this[0, j];
            
            // HUGE OPTIMIZATION: If the element is 0.0, anything multiplied by it is 0.
            // We can skip the entire recursive calculation for this branch!
            if (element != 0.0)
            {
                var minor = GetMinor(0, j);
                determinant += sign * element * minor.Determinant();
            }
            
            // Alternate signs (+, -, +, -)
            sign *= -1;
        }

        return determinant;
    }

    /// <summary>
    /// Checks if the matrix is Upper Triangular (all elements below the main diagonal are zero).
    /// </summary>
    public bool IsUpperTriangular(double tolerance = 1e-14)
    {
        if (!IsSquare) return false;

        for (var i = 1; i < Rows; i++)
        {
            for (var j = 0; j < i; j++)
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
        if (!IsSquare) return false;

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

        var identity = Matrix.Identity(Rows);
        var product = this * this.Transpose();

        return product.IsCloseTo(identity, atol: tolerance);
    }

    /// <summary>
    /// Extracts the elements on the main diagonal into a flat 1D array.
    /// </summary>
    public double[] GetDiagonal()
    {
        int minDim = Math.Min(Rows, Cols);
        double[] diag = new double[minDim];

        for (int i = 0; i < minDim; i++)
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
        if (Data.Length < 100_000)
        {
            var sum = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = 0;

            var vOnes = new Vector<double>(1.0);

            for (; i <= Data.Length - vectorSize; i += vectorSize)
            {
                var v = new Vector<double>(Data, i);
                var vAbs = Vector.Abs(v);
                sum += Vector.Dot(v, vAbs);
            }

            for (; i < Data.Length; i++)
            {
                sum += Math.Abs(Data[i]);
            }

            return sum;
        }

        var globalSum = 0.0;
        var lockObj = new object();
        double[] data = Data;

        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, Data.Length), () => 0.0,
            (range, loopState, localSum) =>
            {
                int vectorSize = Vector<double>.Count;
                int i = range.Item1;
                var vOnes = new Vector<double>(1.0);

                for (; i <= range.Item2 - vectorSize; i += vectorSize)
                {
                    var v = new Vector<double>(data, i);
                    localSum += Vector.Dot(Vector.Abs(v), vOnes);
                }

                for (; i < range.Item2; i++) localSum += Math.Abs(data[i]);

                return localSum;
            },
            (localSum) => { lock (lockObj) globalSum += localSum; }
        );

        return globalSum;
    }

    /// <summary>
    /// Calculates the L2 Norm (Euclidean Magnitude) of the matrix/vector.
    /// Also known mathematically as the Frobenius Norm for 2D matrices.
    /// </summary>
    private double NormL2()
    {
        if (Data.Length < 100_000)
        {
            var sumOfSquares = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = 0;

            for (; i <= Data.Length - vectorSize; i += vectorSize)
            {
                var v = new Vector<double>(Data, i);
                sumOfSquares += Vector.Dot(v, v);
            }

            for (; i < Data.Length; i++)
            {
                sumOfSquares += Data[i] * Data[i];
            }

            return Math.Sqrt(sumOfSquares);
        }
        
        // Multithread optimization
        var globalSum = 0.0;
        var lockObj = new object();
        double[] currentData = Data;

        Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, Data.Length), () => 0.0, (range, loopState, localSum) =>
            {
                int vectoeSize = Vector<double>.Count;
                int i = range.Item1;

                for (; i <= range.Item2 - vectoeSize; i += vectoeSize)
                {
                    var v = new Vector<double>(currentData, i);
                    localSum += Vector.Dot(v, v);
                }

                for (; i < range.Item2; i++)
                {
                    localSum += currentData[i] * currentData[i];
                }

                return localSum;
            }, 
            (localSum) =>
            {
                lock (lockObj) globalSum += localSum;
            }
        );
        
        return Math.Sqrt(globalSum);
    }

    #endregion
}