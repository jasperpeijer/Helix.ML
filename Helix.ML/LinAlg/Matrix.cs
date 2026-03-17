using System.Numerics;
using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

/// <summary>
/// A high-performance, contiguous-memory representation of a 2D matrix.
/// </summary>
public readonly struct Matrix
{
    public int Rows { get; }
    public int Cols { get; }
    public (int Rows, int Cols) Shape => (Rows, Cols);

    public readonly double[] Data;

    #region Constructors
    
    /// <summary>
    /// Standard constructor using separate row/col arguments.
    /// </summary>
    public Matrix(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Matrix dimensions must be strictly positive.");
        }
        
        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];
    }

    /// <summary>
    /// ML-style constructor using a Shape tuple. 
    /// Calls the standard constructor via 'this()'.
    /// </summary>
    public Matrix((int rows, int cols) shape) : this(shape.rows, shape.cols) {}

    /// <summary>
    /// Initializes a matrix from an existing flat array.
    /// </summary>
    public Matrix(double[,] data)
    {
        var rows = data.GetLength(0);
        var cols = data.GetLength(1);

        if (rows == 0 || cols == 0)
        {
            throw new ArgumentException("Matrix dimensions must be strictly positive.");
        }

        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];

        // The High-Performance Memory Copy
        // A double takes up 8 bytes of memory. We calculate the total bytes needed 
        // and copy the raw memory block directly, bypassing all bounds-checking loops.
        var totalBytes = rows * cols * sizeof(double);
        Buffer.BlockCopy(data, 0, Data, 0, totalBytes);
    }

    /// <summary>
    /// Initializes a matrix from an existing flat 1D array.
    /// </summary>
    public Matrix(int rows, int cols, double[] data)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Matrix dimensions must be strictly positive.");
        }

        if (data.Length != rows * cols)
        {
            throw new ArgumentException("Data length does not match the provided matrix dimensions.");
        }
        
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    /// <summary>
    /// Initializes a matrix from an existing flat 1D array, with a tuple passed for the dimensions
    /// </summary>
    public Matrix((int rows, int cols) shape, double[] data) : this(shape.rows, shape.cols, data) {}
    
    #endregion

    #region Matrix Operations
    
    /// <summary>
    /// Multiplies a matrix by a scalar value using SIMD.
    /// </summary>
    public static Matrix operator *(Matrix m, double scalar)
    {
        double[] result = new double[m.Data.Length];
        int i = 0;
        int vectorSize = Vector<double>.Count;
        
        // Broadcast the single scalar value into a full hardware vector
        var vScalar = new Vector<double>(scalar);
        
        for (; i < m.Data.Length - vectorSize; i += vectorSize)
        {
            var va = new Vector<double>(m.Data, i);
            var vResult = va * vScalar;
            vResult.CopyTo(result, i);
        }

        for (; i < m.Data.Length; i++)
        {
            result[i] = m.Data[i] * scalar;
        }
        
        return new Matrix(m.Shape, result);
    }
    
    /// <summary>
    /// Supports the commutative property (e.g., 5.0 * matrix instead of matrix * 5.0).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator *(double scalar, Matrix m) => m * scalar;

    /// <summary>
    /// Adds a scalar value to every element in the matrix using SIMD.
    /// </summary>
    public static Matrix operator +(Matrix m, double scalar)
    {
        var result = new double[m.Data.Length];
        var vectorSize = Vector<double>.Count;
        var vScalar = new Vector<double>(scalar);
        int i = 0;

        for (; i <= m.Data.Length - vectorSize; i += vectorSize)
        {
            var vA = new Vector<double>(m.Data, i);
            var vRes = vA + vScalar;
            vRes.CopyTo(result, i);
        }

        for (; i < m.Data.Length; i++)
        {
            result[i] = m.Data[i] + scalar;
        }
        
        return new Matrix(m.Shape, result);
    }
    
    /// <summary>
    /// Supports the commutative property (e.g., 5.0 + matrix).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator +(double scalar, Matrix m) => m + scalar;
    
    /// <summary>
    /// Adds two matrices together using SIMD
    /// </summary>
    public static Matrix operator +(Matrix m1, Matrix m2)
    {
        if (m1.Shape != m2.Shape)
        {
            throw new ArgumentException($"Matrix shapes must match for addition. Got {m1.Shape} and {m2.Shape}.");
        }
        
        var result = new double[m1.Data.Length];
        var i = 0;
        
        // Vector<double>.Count dynamically checks the CPU at runtime to see 
        // how many doubles it can process simultaneously (usually 4 or 8).
        var vectorSize = Vector<double>.Count;
        
        // The SIMD Loop: Process data in massive chunks
        // We stop right before we run out of enough elements to fill a full vector
        for (; i < m1.Data.Length - vectorSize; i += vectorSize)
        {
            // Load chunks from RAM into CPU registers
            var va = new Vector<double>(m1.Data, i);
            var vb = new Vector<double>(m2.Data, i);

            // The hardware adds all 4 or 8 numbers in a single clock cycle
            var vResult = va + vb;
            
            // Push the chunk back to RAM
            vResult.CopyTo(result, i);
        }
        
        // The Tail Loop: Catch the leftovers
        // If our array has 10 elements and the CPU processes 4 at a time,
        // elements 8 and 9 are left over. We process them normally.
        for (; i < m1.Data.Length; i++)
        {
            result[i] = m1.Data[i] + m2.Data[i];
        }
        
        return new Matrix(m1.Shape, result);
    }

    /// <summary>
    /// Increments every element in the matrix by 1.0 using SIMD hardware acceleration.
    /// </summary>
    public static Matrix operator ++(Matrix m)
    {
        var result = new Matrix(m.Rows, m.Cols);
        int vectorSize = Vector<double>.Count;
        int i = 0;

        var vOne = new Vector<double>(1.0);

        for (; i <= m.Data.Length - vectorSize; i += vectorSize)
        {
            var vM = new Vector<double>(m.Data, i);
            var vRes = vM + vOne;
            vRes.CopyTo(result.Data, i);
        }

        for (; i < m.Data.Length; i++)
        {
            result.Data[i] = m.Data[i] + 1.0;
        }

        return result;
    }

    /// <summary>
    /// Decrements every element in the matrix by 1.0 using SIMD hardware acceleration.
    /// </summary>
    public static Matrix operator --(Matrix m)
    {
        var result = new Matrix(m.Rows, m.Cols);
        var vectorSize = Vector<double>.Count;
        int i = 0;
        var vOne = new Vector<double>(1.0);

        for (; i <= m.Data.Length - vectorSize; i += vectorSize)
        {
            var vM = new Vector<double>(m.Data, i);
            var vRes = vM - vOne;
            vRes.CopyTo(result.Data, i);
        }

        for (; i < m.Data.Length; i++)
        {
            result.Data[i] = m.Data[i] - 1.0;
        }

        return result;
    }

    /// <summary>
    /// Unary negation: Flips the sign of every element in the matrix (-A).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator -(Matrix m) => m * (-1);

    /// <summary>
    /// Subtracts a scalar value from every element in the matrix using SIMD.
    /// </summary>
    public static Matrix operator -(Matrix m, double scalar) => m + (-scalar);
    
    /// <summary>
    /// Subtracts matrix b from matrix a element-wise using SIMD.
    /// </summary>
    public static Matrix operator -(Matrix m1, Matrix m2)
    {
        if (m1.Shape != m2.Shape)
        {
            throw new ArgumentException($"Matrix shapes must match for addition. Got {m1.Shape} and {m2.Shape}.");
        }
        
        var result = new double[m1.Data.Length];
        var i = 0;
        
        // Vector<double>.Count dynamically checks the CPU at runtime to see 
        // how many doubles it can process simultaneously (usually 4 or 8).
        var vectorSize = Vector<double>.Count;
        
        // The SIMD Loop: Process data in massive chunks
        // We stop right before we run out of enough elements to fill a full vector
        for (; i < m1.Data.Length - vectorSize; i += vectorSize)
        {
            // Load chunks from RAM into CPU registers
            var va = new Vector<double>(m1.Data, i);
            var vb = new Vector<double>(m2.Data, i);

            // The hardware adds all 4 or 8 numbers in a single clock cycle
            var vResult = va - vb;
            
            // Push the chunk back to RAM
            vResult.CopyTo(result, i);
        }
        
        // The Tail Loop: Catch the leftovers
        // If our array has 10 elements and the CPU processes 4 at a time,
        // elements 8 and 9 are left over. We process them normally.
        for (; i < m1.Data.Length; i++)
        {
            result[i] = m1.Data[i] - m2.Data[i];
        }
        
        return new Matrix(m1.Shape, result);
    }

    /// <summary>
    /// Performs high-performance Matrix-Matrix Multiplication (Dot Product).
    /// Utilizes loop reordering, SIMD vectorization, and multi-core parallelism.
    /// </summary>
    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
        {
            throw new ArgumentException($"Inner dimensions must match. Got A: {a.Shape} and B: {b.Shape}.");
        }

        var result = new Matrix(a.Rows, b.Cols);
        int vectorSize = Vector<double>.Count;

        Parallel.For(0, a.Rows, i =>
        {
            int cRowOffset = i * result.Cols;

            for (int k = 0; k < a.Cols; k++)
            {
                double a_ik = a.Data[i * a.Cols + k];

                if (a_ik == 0) continue;
                
                var vA = new Vector<double>(a_ik);
                int bRowOffset = k * b.Cols;
                int j = 0;

                for (; j <= b.Cols - vectorSize; j += vectorSize)
                {
                    var vB = new Vector<double>(b.Data, bRowOffset + j);
                    var vC = new Vector<double>(result.Data, cRowOffset + j);

                    var vRes = vC + (vA * vB);
                    
                    vRes.CopyTo(result.Data, cRowOffset + j);
                }

                for (; j < b.Cols; j++)
                {
                    result.Data[cRowOffset + j] += a_ik * b.Data[bRowOffset + j];
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Divides every element in the matrix by a scalar value by multiplying by its reciprocal.
    /// </summary>
    public static Matrix operator /(Matrix a, double scalar)
    {
        if (scalar == 0)
        {
            throw new DivideByZeroException("Cannot divide a matrix by zero.");
        }
        
        return a * (1.0/scalar);
    }

    /// <summary>
    /// Horizontally concatenates another matrix to the right side of this matrix.
    /// Useful for creating augmented matrices [A | B].
    /// </summary>
    public Matrix Augment(Matrix right)
    {
        if (this.Rows != right.Rows)
        {
            throw new ArgumentException("Matrices must have the exact same number of rows to be augmented.");
        }
        
        var result = new Matrix(this.Rows, this.Cols + right.Cols);

        result[.., 0..this.Cols] = this;
        result[.., this.Cols..] = right;

        return result;
    }

    /// <summary>
    /// Syntactic sugar for Augmenting two matrices [A | B].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator |(Matrix left, Matrix right) => left.Augment(right);

    /// <summary>
    /// Vertically concatenates another matrix to the bottom of this matrix.
    /// Useful for stacking datasets or batches.
    /// </summary>
    public Matrix Concatenate(Matrix bottom)
    {
        if (this.Cols != bottom.Cols)
        {
            throw new ArgumentException("Matrices must have the exact same number of columns to be concatenated vertically.");
        }
        
        var result = new Matrix(this.Rows + bottom.Rows, this.Cols);

        Array.Copy(this.Data, 0, result.Data, 0, this.Data.Length);
        Array.Copy(bottom.Data, 0, result.Data, this.Data.Length, bottom.Data.Length);

        return result;
    }
    
    /// <summary>
    /// Syntactic sugar for Concatenating two matrices vertically.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Matrix operator &(Matrix top, Matrix bottom) => top.Concatenate(bottom);
    
    #endregion
    
    #region Special Matrices
    
    /// <summary>
    /// Creates a matrix of the specified dimensions filled entirely with zeros.
    /// </summary>
    public static Matrix Zeros(int rows, int cols)
    {
        return new Matrix(rows, cols);
    }
    
    /// <summary>
    /// Creates a matrix of the specified shape filled entirely with zeros.
    /// </summary>
    public static Matrix Zeros((int rows, int cols) shape)
    {
        return new Matrix(shape.rows, shape.cols);
    }

    /// <summary>
    /// Creates an Identity matrix. If the dimensions are unequal, it places 1.0s on the main diagonal 
    /// until the smallest dimension is exhausted.
    /// </summary>
    public static Matrix Identity(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
        {
            throw new ArgumentException("Matrix dimensions must be strictly positive.");    
        }

        var identityMatrix = new Matrix(rows, cols);
        
        int minDim = Math.Min(rows, cols);

        for (int i = 0; i < minDim; i++)
        {
            identityMatrix.Data[(i * cols) + i] = 1.0;
        }

        return identityMatrix;
    }
    
    /// <summary>
    /// Creates an Identity matrix using tuple. If the dimensions are unequal, it places 1.0s on the main diagonal 
    /// until the smallest dimension is exhausted.
    /// </summary>
    public static Matrix Identity((int rows, int cols) shape) => Identity(shape.rows, shape.cols);
    
    /// <summary>
    /// Creates a square N x N Identity matrix.
    /// </summary>
    public static Matrix Identity(int size) => Identity(size, size);

    /// <summary>
    /// Creates a matrix of the specified dimensions filled entirely with ones.
    /// </summary>
    public static Matrix Ones(int rows, int cols)
    {
        var matrix = new Matrix(rows, cols);
        Array.Fill(matrix.Data, 1.0);

        return matrix;
    }
    
    /// <summary>
    /// Creates a matrix of the specified shape filled entirely with ones.
    /// </summary>
    public static Matrix Ones((int rows, int cols) shape) => Ones(shape.rows, shape.cols);

    /// <summary>
    /// Creates a matrix filled with uniformly distributed random doubles between min and max.
    /// Default range is 0.0 to 1.0.
    /// </summary>
    public static Matrix Random(int rows, int cols, double min = 0.0, double max = 1.0)
    {
        var matrix = new Matrix(rows, cols);

        for (int i = 0; i < matrix.Data.Length; i++)
        {
            matrix.Data[i] = min + (System.Random.Shared.NextDouble() * (max - min));
        }

        return matrix;
    }
    
    /// <summary>
    /// Creates a matrix filled with uniformly distributed random doubles between 0.0 and 1.0.
    /// </summary>
    public static Matrix Random((int rows, int cols) shape) => 
        Random(shape.rows, shape.cols);
    
    /// <summary>
    /// Creates a matrix filled with uniformly distributed random doubles between min and max.
    /// </summary>
    public static Matrix Random((int rows, int cols) shape, double min, double max) => 
        Random(shape.rows, shape.cols, min, max);
    
    /// <summary>
    /// Creates a matrix filled with uniformly distributed random doubles between min and max.
    /// </summary>
    public static Matrix Random((int rows, int cols) shape, (double min, double max) range) => 
        Random(shape.rows, shape.cols, range.min, range.max);

    /// <summary>
    /// Creates a matrix filled with random integers (stored as doubles) 
    /// between min (inclusive) and max (exclusive).
    /// </summary>
    public static Matrix RandomInt(int rows, int cols, int min, int max)
    {
        var matrix = new Matrix(rows, cols);

        for (int i = 0; i < matrix.Data.Length; i++)
        {
            matrix.Data[i] = System.Random.Shared.Next(min, max);
        }

        return matrix;
    }
    
    /// <summary>
    /// Creates a matrix filled with random integers (stored as doubles) 
    /// between min (inclusive) and max (exclusive).
    /// </summary>
    public static Matrix RandomInt((int rows, int cols) shape, (int min, int max) range) => 
        RandomInt(shape.rows, shape.cols, range.min, range.max);

    /// <summary>
    /// Returns a new matrix that is the transpose of the current matrix.
    /// Rows become columns, and columns become rows.
    /// </summary>
    public Matrix Transpose()
    {
        int rows = this.Rows;
        int cols = this.Cols;
        double[] originalData = this.Data;
        
        var result = new Matrix(cols, rows);

        Parallel.For(0, Rows, i =>
        {
            int originalRowOffset = i * cols;

            for (var j = 0; j < cols; j++)
            {
                double value = originalData[originalRowOffset + j];
                result.Data[j * result.Cols + i] = value;
            }
        });

        return result;
    }
    
    /// <summary>
    /// Gets the transpose of this matrix (Syntactic Sugar for .Transpose()).
    /// WARNING: This performs a full O(N) memory allocation and multithreaded copy. 
    /// Do not call this repeatedly inside tight loops; cache the result instead.
    /// </summary>
    public Matrix T => Transpose();
    
    #endregion
    
    #region Matrix Properties

    /// <summary>
    /// Calculates the Trace of the matrix (the sum of the main diagonal elements).
    /// </summary>
    public double Trace(bool allowRectangular = false)
    {
        if (!allowRectangular && Rows != Cols)
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
        if (Rows != Cols)
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
        if (Rows != Cols) return false;

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
        if (Rows != Cols) return false;

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
            for (var j = 0; j < Cols; j++)
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
            for (var j = 0; j < Cols; j++)
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
    
    #endregion
    
    public double this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Data[(row * Cols) + col];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => Data[(row * Cols) + col] = value;
    }

    /// <summary>
    /// Extracts a sub-matrix using C# Range syntax (e.g., matrix[0..2, 1..^1]).
    /// </summary>
    public Matrix this[Range rowRange, Range colRange]
    {
        get
        {
            var (rowOffset, rowLength) = rowRange.GetOffsetAndLength(Rows);
            var (colOffset, colLength) = colRange.GetOffsetAndLength(Cols);
            
            var result = new Matrix(rowLength, colLength);

            for (int i = 0; i < rowLength; i++)
            {
                int srcIndex = ((rowOffset + i) * Cols) + colOffset;
                int dstIndex = i * colLength;
                
                Array.Copy(this.Data, srcIndex, result.Data, dstIndex, colLength);
            }

            return result;
        }

        set
        {
            var (rowOffset, rowLength) = rowRange.GetOffsetAndLength(Rows);
            var (colOffset, colLength) = colRange.GetOffsetAndLength(Cols);

            if (value.Rows != rowLength || value.Cols != colLength)
            {
                throw new ArgumentException($"Assigned matrix shape {value.Shape} does not match slice shape ({rowLength}, {colLength}).");
            }

            for (var i = 0; i < rowLength; i++)
            {
                int srcIndex = i * value.Cols;
                int dstIndex = ((rowOffset + i) * Cols) + colOffset;
                
                Array.Copy(value.Data, srcIndex, this.Data, dstIndex, colLength);
            }
        }
    }

    #region Matrix Comparison
    
    public static bool operator ==(Matrix left, Matrix right)
    {
        if (left.Shape != right.Shape) return false;

        const double epsilon = 1e-14;

        for (int i = 0; i < left.Data.Length; i++)
        {
            if (System.Math.Abs(left.Data[i] - right.Data[i]) > epsilon)
            {
                return false;
            }
        }

        return true;
    }
    
    public static bool operator !=(Matrix left, Matrix right) => !(left == right);

    public override bool Equals(object? obj) => obj is Matrix other && this == other;

    public override int GetHashCode()
    {
        int hash = HashCode.Combine(Rows, Cols);
        if (Data.Length > 0) hash = HashCode.Combine(hash, Data[0]);
        if (Data.Length > 1) hash = HashCode.Combine(hash, Data[^1]);

        return hash;
    }

    /// <summary>
    /// Checks if all elements in this matrix are mathematically close to the corresponding elements 
    /// in another matrix, using both relative and absolute tolerances. (Mimics NumPy's np.allclose).
    /// </summary>
    /// <param name="other">The matrix to compare against.</param>
    /// <param name="rtol">The relative tolerance parameter (default 1e-5).</param>
    /// <param name="atol">The absolute tolerance parameter (default 1e-8).</param>
    public bool IsCloseTo(Matrix other, double rtol = 1e-5, double atol = 1e-8)
    {
        if (Shape != other.Shape)
        {
            throw new ArgumentException($"Cannot compare matrices of different shapes. Got {Shape} and {other.Shape}.");
        }

        for (int i = 0; i < Data.Length; i++)
        {
            double a = Data[i];
            double b = other.Data[i];

            double allowedDifference = atol + (rtol * Math.Abs(b));

            if (Math.Abs(a - b) > allowedDifference) return false;
        }

        return true;
    }
    
    #endregion
}