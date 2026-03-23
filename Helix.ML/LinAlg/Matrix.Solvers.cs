using System.Numerics;
using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
    #region Solvers
    
    /// <summary>
    /// Calculates the Inverse analytically using high-performance PLU Decomposition.
    /// </summary>
    public Matrix Inverse(double tolerance = 1e-14)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Only square matrices can be inverted.");

        if (Rows == 1) return new Matrix(1, 1, [1.0 / Data[0]]);

        if (Rows == 2)
        {
            var det = Determinant();
            
            if (Math.Abs(det) < tolerance) 
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            
            var invDet = 1.0 / det;
            
            return new Matrix(2, 2, [
                Data[3] * invDet, -Data[1] * invDet,
                -Data[2] * invDet, Data[0] * invDet
            ]);
        }

        try
        {
            return Solve(Identity(Rows));
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("singular"))
        {
            throw new InvalidOperationException("Matrix is singular and cannot be inverted.", ex);
        }
    }

    /// <summary>
    /// Calculates the Inverse using high-performance PLU Decomposition.
    /// Utilizes a pre-computed determinant to instantly fast-fail singular matrices and optimize the 2x2 path.
    /// </summary>
    public Matrix Inverse(double knownDeterminant, double tolerance = 1e-14)
    {
        if (Math.Abs(knownDeterminant) < tolerance)
            throw new InvalidOperationException("Matrix is singular (determinant is 0) and cannot be inverted.");
        
        if (Rows == 2)
        {
            var invDet = 1.0 / knownDeterminant;
            return new Matrix(2, 2, [
                Data[3] * invDet, -Data[1] * invDet,
                -Data[2] * invDet, Data[0] * invDet
            ]);
        }
        
        return Inverse(tolerance);
    }

    /// <summary>
    /// Syntactic sugar for Matrix Inverse (!A).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator !(Matrix matrix) => matrix.Inverse();

    /// <summary>
    /// Calculates the Moore-Penrose Pseudoinverse using Singular Value Decomposition (SVD).
    /// Automatically and safely handles square, tall, wide, and rank-deficient matrices.
    /// </summary>
    public Matrix PseudoInverse(double tolerance = 1e-14)
    {
        var (u, s, v) = SVD();
        var sPlus = Zeros(s.Cols, s.Rows);
        var minDim = Math.Min(s.Rows, s.Cols);

        for (var i = 0; i < minDim; i++)
        {
            if (s[i, i] > tolerance)
            {
                sPlus[i, i] = 1.0 / s[i, i];
            }
        }

        return v * sPlus * u.T;
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

    /// <summary>
    /// Performs PLU Decomposition (Partial Pivoting Gaussian Elimination).
    /// Factors a matrix into P * A = L * U. 
    /// Gracefully handles singular matrices and utilizes multithreading for massive datasets.
    /// </summary>
    // ReSharper disable once InconsistentNaming
    public (int[] P, Matrix L, Matrix U, int Swaps) PLUDecomposition(double tolerance = 1e-14)
    {
        // Not fully sure how this algorithm works, but it's a computed version
        // of the Gaussian elimination method for PLU decomposition.
        
        if (!IsSquare)
            throw new InvalidOperationException("LU Decomposition is only defined for square matrices.");

        var n = Rows;
        var swaps = 0;
        
        // U starts as a direct clone of A. L starts as an Identity matrix.
        var uData = new double[Data.Length];
        Array.Copy(Data, uData, Data.Length);
        var u = new Matrix(n, n, uData);
        var l = Matrix.Identity(n);

        var p = new int[n];
        for (var i = 0; i < n; i++) p[i] = i;

        for (var i = 0; i < n; i++)
        {
            var pivotRow = i;
            var maxVal = Math.Abs(u[i, i]);

            for (var k = i + 1; k < n; k++)
            {
                if (Math.Abs(u[k, i]) > maxVal)
                {
                    maxVal = Math.Abs(u[k, i]);
                    pivotRow = k;
                }
            }

            if (maxVal < tolerance) continue;

            if (pivotRow != i)
            {
                for (var col = 0; col < n; col++)
                {
                    (u[i, col], u[pivotRow, col]) = (u[pivotRow, col], u[i, col]);
                }

                for (var col = 0; col < i; col++)
                {
                    (l[i, col], l[pivotRow, col]) = (l[pivotRow, col], l[i, col]);
                }

                (p[i], p[pivotRow]) = (p[pivotRow], p[i]);

                swaps++;
            }
            
            // Multithread Gaussian Elimination
            var remainingRows = n - (i + 1);

            if (remainingRows > 128)
            {
                Parallel.For(i + 1, n, k =>
                {
                    var factor = u[k, i] / u[i, i];
                    l[k, i] = factor;

                    for (var j = i; j < n; j++)
                    {
                        u[k, j] -= factor * u[i, j];
                    }
                });
            }
            else
            {
                for (var k = i + 1; k < n; k++)
                {
                    var factor = u[k, i] / u[i, i];
                    l[k, i] = factor;

                    for (var j = i; j < n; j++)
                    {
                        u[k, j] -= factor * u[i, j];
                    }
                }
            }
        }

        return (p, l, u, swaps);
    }

    /// <summary>
    /// Performs QR Decomposition using Householder Reflections and SIMD Intrinsics.
    /// Utilizes "The Transpose Trick" to force contiguous memory alignment for AVX registers.
    /// </summary>
    // ReSharper disable once InconsistentNaming
    // Implemented the Gram-Schmidt way first, but that is numerically unstable, so changed to
    // householder reflections. Could not be bothered to understand this algorithm since it's
    // complicated, especially with SIMD optimization. (Thank you Gemini)
    public (Matrix Q, Matrix R) QRDecomposition()
    {
        var (m, n) = Shape;
        var rT = Transpose(); 
        var q = Identity(m);
        var loops = Math.Min(m, n);
        var vectorSize = Vector<double>.Count;

        for (var k = 0; k < loops; k++)
        {
            var normX = 0.0;
            
            for (var i = k; i < m; i++) normX += rT[k, i] * rT[k, i]; 
            
            normX = Math.Sqrt(normX);

            if (normX < 1e-14) continue; 

            double s = Math.Sign(rT[k, k]);
            
            if (s == 0) s = 1.0;

            var u1 = rT[k, k] + s * normX;
            var v = new double[m - k];
            v[0] = 1.0;
            var normV = 1.0;
            
            for (var i = 1; i < m - k; i++)
            {
                v[i] = rT[k, k + i] / u1;
                normV += v[i] * v[i];
            }

            var tau = 2.0 / normV;
            
            for (var j = k; j < n; j++)
            {
                var sum = 0.0;
                var i = 0;
                var rTOffset = j * rT.Cols + k;
                
                for (; i <= (m - k) - vectorSize; i += vectorSize)
                {
                    var vecV = new Vector<double>(v, i);
                    var vecR = new Vector<double>(rT.Data, rTOffset + i);
                    sum += Vector.Dot(vecV, vecR);
                }

                for (; i < m - k; i++) sum += v[i] * rT.Data[rTOffset + i];

                var mult = tau * sum;
                i = 0;
                var vecMult = new Vector<double>(mult);
                
                for (; i <= (m - k) - vectorSize; i += vectorSize)
                {
                    var vecV = new Vector<double>(v, i);
                    var vecR = new Vector<double>(rT.Data, rTOffset + i);
                    (vecR - (vecMult * vecV)).CopyTo(rT.Data, rTOffset + i);
                }
                
                for (; i < m - k; i++) rT.Data[rTOffset + i] -= mult * v[i];
            }
            
            for (var i = k + 1; i < m; i++) rT[k, i] = 0.0;
            
            for (var i = 0; i < m; i++)
            {
                var sum = 0.0;
                var j2 = 0;
                var qOffset = i * q.Cols + k;

                for (; j2 <= (m - k) - vectorSize; j2 += vectorSize)
                {
                    var vecV = new Vector<double>(v, j2);
                    var vecQ = new Vector<double>(q.Data, qOffset + j2);
                    sum += Vector.Dot(vecQ, vecV);
                }
                
                for (; j2 < m - k; j2++) sum += q.Data[qOffset + j2] * v[j2];

                var mult = tau * sum;
                j2 = 0;
                var vecMult = new Vector<double>(mult);
                
                for (; j2 <= (m - k) - vectorSize; j2 += vectorSize)
                {
                    var vecV = new Vector<double>(v, j2);
                    var vecQ = new Vector<double>(q.Data, qOffset + j2);
                    (vecQ - (vecMult * vecV)).CopyTo(q.Data, qOffset + j2);
                }
                
                for (; j2 < m - k; j2++) q.Data[qOffset + j2] -= mult * v[j2];
            }
        }
        
        return (q, rT.Transpose()); 
    }
    
    /// <summary>
    /// Reduces a square matrix to Upper Hessenberg form using double-sided Householder reflections.
    /// For symmetric matrices, this naturally produces a Tridiagonal matrix, massively accelerating Eigendecomposition.
    /// </summary>
    /// Could not be bothered to learn this algorithm
    public (Matrix Q, Matrix H) HessenbergReduction()
    {
        if (!IsSquare) 
            throw new InvalidOperationException("Hessenberg reduction requires a square matrix.");
            
        var n = Rows;
        
        var hData = new double[Data.Length];
        Array.Copy(Data, hData, Data.Length);
        var h = new Matrix(n, n, hData);
        
        var q = Identity(n);

        for (var k = 0; k < n - 2; k++)
        {
            double normX = 0.0;
            for (int i = k + 1; i < n; i++) normX += h[i, k] * h[i, k];
            normX = Math.Sqrt(normX);

            if (normX < 1e-14) continue;

            double s = Math.Sign(h[k + 1, k]);
            if (s == 0) s = 1.0;

            var u1 = h[k + 1, k] + s * normX;
            var v = new double[n - (k + 1)];
            v[0] = 1.0;
            var normV = 1.0;
            
            for (var i = 1; i < v.Length; i++)
            {
                v[i] = h[k + 1 + i, k] / u1;
                normV += v[i] * v[i];
            }

            var tau = 2.0 / normV;

            for (var j = k; j < n; j++)
            {
                var sum = 0.0;
                for (var i = 0; i < v.Length; i++) sum += v[i] * h[k + 1 + i, j];
                
                var mult = tau * sum;
                for (var i = 0; i < v.Length; i++) h[k + 1 + i, j] -= mult * v[i];
            }
            
            for (var i = 0; i < n; i++)
            {
                var sum = 0.0;
                for (var j = 0; j < v.Length; j++) sum += h[i, k + 1 + j] * v[j];
                
                var mult = tau * sum;
                for (var j = 0; j < v.Length; j++) h[i, k + 1 + j] -= mult * v[j];
            }

            for (var i = 0; i < n; i++)
            {
                var sum = 0.0;
                for (var j = 0; j < v.Length; j++) sum += q[i, k + 1 + j] * v[j];
                
                var mult = tau * sum;
                for (var j = 0; j < v.Length; j++) q[i, k + 1 + j] -= mult * v[j];
            }
            
            for (var i = k + 2; i < n; i++) h[i, k] = 0.0;
        }

        return (q, h);
    }

    /// <summary>
    /// Calculates the real Eigenvalues of a square matrix using the iterative QR algorithm.
    /// </summary>
    public double[] Eigenvalues(int maxIteration = 1000, double tolerance = 1e-14)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Eigenvalues are only defined for square matrices.");

        var (_, currentA) = HessenbergReduction();

        for (var i = 0; i < maxIteration; i++)
        {
            var (q, r) = currentA.QRDecomposition();
            var nextA = r * q;

            if (nextA.IsUpperTriangular(tolerance))
            {
                return nextA.GetDiagonal();
            }

            currentA = nextA;
        }
        
        throw new TimeoutException("QR Algorithm failed to converge. The matrix may contain complex eigenvalues.");
    }

    /// <summary>
    /// Performs Eigendecomposition on a symmetric square matrix.
    /// Returns the Eigenvectors as the columns of Q, and the Eigenvalues as a flat array.
    /// </summary>
    public (Matrix Eigenvectors, double[] Eigenvalues) EigenDecomposition(int maxIterations = 1000,
        double tolerance = 1e-14)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Eigendecomposition is only defined for square matrices.");
        if (!IsSymmetric())
            throw new InvalidOperationException("For ML stability, this engine only decomposes symmetric matrices.");
        
        var (totalQ, currentA) = HessenbergReduction();
        
        for (var i = 0; i < maxIterations; i++)
        {
            var (q, r) = currentA.QRDecomposition();
            
            totalQ *= q;
            var nextA = r * q; 
            
            if (nextA.IsUpperTriangular(tolerance))
            {
                return (totalQ, nextA.GetDiagonal());
            }
            
            currentA = nextA;
        }
        
        throw new TimeoutException("QR Algorithm failed to converge.");
    }

    /// <summary>
    /// Performs Reduced (Thin) Singular Value Decomposition (SVD).
    /// Factors any matrix into A = U * S * V^T.
    /// Unconditionally stable for all matrix shapes and ranks.
    /// </summary>
    public (Matrix U, Matrix S, Matrix V) SVD(int maxIterations = 1000, double tolerance = 1e-9)
    {
        // K is the maximum possible number of true stretching axes
        int k = Math.Min(Rows, Cols);
        
        // 1. Calculate A^T * A to get the Right Singular Vectors (V)
        var aTa = this.T * this;
        var (fullV, eigenvalues) = aTa.EigenDecomposition(maxIterations, tolerance);

        // 2. Extract and sort Singular Values descending
        var singularPairs = new List<(double Value, int OriginalIndex)>();
        for (int i = 0; i < eigenvalues.Length; i++)
        {
            double sv = eigenvalues[i] > tolerance ? Math.Sqrt(eigenvalues[i]) : 0.0;
            singularPairs.Add((sv, i));
        }
        singularPairs.Sort((a, b) => b.Value.CompareTo(a.Value));

        // 3. Initialize the strictly reduced shapes!
        var s = Zeros(k, k);
        var v = new Matrix(Cols, k);
        var u = new Matrix(Rows, k); 

        for (int j = 0; j < k; j++) // Only keep the top K components
        {
            var pair = singularPairs[j];
            
            // Populate the diagonal of S
            s[j, j] = pair.Value;

            // Map the corresponding V vector
            for (int i = 0; i < Cols; i++) v[i, j] = fullV[i, pair.OriginalIndex];

            // Map the U vector using U_i = (A * V_i) / sigma_i
            if (pair.Value > tolerance)
            {
                for (int i = 0; i < Rows; i++)
                {
                    double dot = 0.0;
                    for (int c = 0; c < Cols; c++) dot += this[i, c] * fullV[c, pair.OriginalIndex];
                    u[i, j] = dot / pair.Value;
                }
            }
        }

        return (u, s, v);
    }

    /// <summary>
    /// Solves a system of linear equations (Ax = b) using PLU Decomposition.
    /// Can solve for a single vector or multiple vectors simultaneously.
    /// </summary>
    public Matrix Solve(Matrix b, double tolerance=1e-14)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Matrix A must be square to solve Ax=b.");
        
        if (Rows != b.Rows)
            throw new ArgumentException($"The number of rows in b ({b.Rows}) must match A ({Rows}).");
        
        var (p, l, u, _) = PLUDecomposition();

        var pb = new Matrix(Rows, b.Cols);

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < b.Cols; j++)
            {
                pb[i, j] = b[p[i], j];
            }
        }

        var y = new Matrix(Rows, b.Cols);

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < b.Cols; j++)
            {
                var sum = 0.0;
                
                for (var k = 0; k < i; k++) sum += l[i, k] * y[k, j];
                
                y[i, j] = pb[i, j] - sum;
            }
        }
        
        var x = new Matrix(Rows, b.Cols);

        for (var i = Rows - 1; i >= 0; i--)
        {
            for (var j = 0; j < b.Cols; j++)
            {
                var sum = 0.0;
                
                for (var k = i + 1; k < Rows; k++) sum += u[i, k] * x[k, j];
                
                if (Math.Abs(u[i, i]) < tolerance) 
                    throw new InvalidOperationException("Matrix is singular and cannot be solved.");
                
                x[i, j] = (y[i, j] - sum) / u[i, i];
            }
        }

        return x;
    }
    
    #endregion Solvers
}