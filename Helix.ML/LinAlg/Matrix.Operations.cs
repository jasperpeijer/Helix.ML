using System.Numerics;
using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
    #region Matrix Operations
    
    /// <summary>
    /// Multiplies a matrix by a scalar value using SIMD.
    /// </summary>
    public static Matrix operator *(Matrix m, double scalar)
    {
        var result = new double[m.Data.Length];
        var data = m.Data;

        if (data.Length < 100_000) ProcessChunk(0, data.Length, data, scalar, result);
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length),
                range => { ProcessChunk(range.Item1, range.Item2, data, scalar, result); });
        }

        return new Matrix(m.Shape, result);
        
        static void ProcessChunk(int start, int end, double[] src, double s, double[] dst)
        {
            var vectorSize = Vector<double>.Count;
            var i = start;
            var vScalar = new Vector<double>(s);
            
            for (; i <= end - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(src, i);
                (va * vScalar).CopyTo(dst, i);
            }
            for (; i < end; i++) dst[i] = src[i] * s;
        }
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
        var data = m.Data;

        if (data.Length < 100_000) ProcessChunk(0, data.Length, data, scalar, result);
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length),
                range => { ProcessChunk(range.Item1, range.Item2, data, scalar, result); });
        }

        return new Matrix(m.Shape, result);
        
        static void ProcessChunk(int start, int end, double[] src, double s, double[] dst)
        {
            var vectorSize = Vector<double>.Count;
            var i = start;
            var vScalar = new Vector<double>(s);

            for (; i <= end - vectorSize; i += vectorSize)
            {
                var vA = new Vector<double>(src, i);
                (vA + vScalar).CopyTo(dst, i);
            }
            for (; i < end; i++) dst[i] = src[i] + s;
        }
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
            throw new ArgumentException($"Matrix shapes must match for addition. Got {m1.Shape} and {m2.Shape}.");
        
        var result = new double[m1.Data.Length];
        var data1 = m1.Data;
        var data2 = m2.Data;

        if (data1.Length < 100_000) ProcessChunk(0, data1.Length, data1, data2, result);
        else
        {
            Parallel.ForEach(
                System.Collections.Concurrent.Partitioner.Create(0, data1.Length),
                (range) => { ProcessChunk(range.Item1, range.Item2, data1, data2, result); }
            );
        }
        
        return new Matrix(m1.Shape, result);
        
        static void ProcessChunk(int start, int end, double[] a, double[] b, double[] dst)
        {
            var vectorSize = Vector<double>.Count;
            var i = start;
            for (; i <= end - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                (va + vb).CopyTo(dst, i);
            }
            for (; i < end; i++) dst[i] = a[i] + b[i];
        }
    }

    /// <summary>
    /// Increments every element in the matrix by 1.0 using SIMD hardware acceleration.
    /// </summary>
    public static Matrix operator ++(Matrix m) => m + 1.0;

    /// <summary>
    /// Decrements every element in the matrix by 1.0 using SIMD hardware acceleration.
    /// </summary>
    public static Matrix operator --(Matrix m) => m + (-1.0);

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
            throw new ArgumentException($"Matrix shapes must match for addition. Got {m1.Shape} and {m2.Shape}.");
        
        var result = new double[m1.Data.Length];
        var data1 = m1.Data;
        var data2 = m2.Data;
        
        if (data1.Length < 100_000) ProcessChunk(0, data1.Length, data1, data2, result);
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data1.Length), range =>
            {
                ProcessChunk(range.Item1, range.Item2, data1, data2, result);
            });
        }
        
        return new Matrix(m1.Shape, result);
        
        static void ProcessChunk(int start, int end, double[] a, double[] b, double[] dst)
        {
            var vectorSize = Vector<double>.Count;
            var i = start;
            
            for (; i <= end - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                (va - vb).CopyTo(dst, i);
            }
            
            for (; i < end; i++) dst[i] = a[i] - b[i];
        }
    }

    /// <summary>
    /// Performs high-performance Matrix-Matrix Multiplication (Dot Product).
    /// Utilizes loop reordering, SIMD vectorization, and multicore parallelism.
    /// </summary>
    public static Matrix operator *(Matrix a, Matrix b)
    {
        if (a.Cols != b.Rows)
            throw new ArgumentException($"Inner dimensions must match. Got A: {a.Shape} and B: {b.Shape}.");

        var result = new Matrix(a.Rows, b.Cols);
        var vectorSize = Vector<double>.Count;

        Parallel.For(0, a.Rows, i =>
        {
            var cRowOffset = i * result.Cols;

            for (var k = 0; k < a.Cols; k++)
            {
                var aik = a.Data[i * a.Cols + k];

                if (aik == 0) continue;
                
                var vA = new Vector<double>(aik);
                var bRowOffset = k * b.Cols;
                var j = 0;

                for (; j <= b.Cols - vectorSize; j += vectorSize)
                {
                    var vB = new Vector<double>(b.Data, bRowOffset + j);
                    var vC = new Vector<double>(result.Data, cRowOffset + j);

                    var vRes = vC + (vA * vB);
                    
                    vRes.CopyTo(result.Data, cRowOffset + j);
                }

                for (; j < b.Cols; j++)
                {
                    result.Data[cRowOffset + j] += aik * b.Data[bRowOffset + j];
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
        if (scalar == 0) throw new DivideByZeroException("Cannot divide a matrix by zero.");
        
        return a * (1.0/scalar);
    }

    /// <summary>
    /// Horizontally concatenates another matrix to the right side of this matrix.
    /// Useful for creating augmented matrices [A | B].
    /// </summary>
    public Matrix Augment(Matrix right)
    {
        if (Rows != right.Rows)
            throw new ArgumentException("Matrices must have the exact same number of rows to be augmented.");

        return new Matrix(this.Rows, this.Cols + right.Cols)
        {
            [.., 0..Cols] = this,
            [.., this.Cols..] = right
        };;
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
            throw new ArgumentException("Matrices must have the exact same number of columns to be concatenated vertically.");
        
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

    /// <summary>
    /// Calculates the flat, element-wise Dot Product of two matrices/vectors.
    /// </summary>
    public double DotProduct(Matrix other)
    {
        if (this.Data.Length != other.Data.Length)
            throw new ArgumentException("Matrices must have the same total number of elements to calculate a flat dot product.");
        
        var thisData = this.Data;
        var otherData = other.Data;

        if (Data.Length < 100_000) return ProcessChunk(0, thisData.Length, thisData, otherData);
        else
        {

            var globalSum = 0.0;
            var lockObj = new object();

            Parallel.ForEach(
                System.Collections.Concurrent.Partitioner.Create(0, thisData.Length),
                () => 0.0,
                (range, _, localSum) => localSum + ProcessChunk(range.Item1, range.Item2, thisData, otherData),
                (localSum) =>
                {
                    lock (lockObj) globalSum += localSum;
                }
            );

            return globalSum;
        }

        static double ProcessChunk(int start, int end, double[] a, double[] b)
        {
            var sum = 0.0;
            var vectorSize = Vector<double>.Count;
            var i = start;

            for (; i <= end - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                sum += Vector.Dot(va, vb);
            }
            for (; i < end; i++) sum += a[i] * b[i];
            
            return sum;
        }
    }

    /// <summary>
    /// Computes the Hadamard Product (element-wise multiplication) of two matrices.
    /// Used heavily in neural network activation derivatives and dropout masks.
    /// </summary>
    public Matrix HadamardProduct(Matrix other)
    {
        if (Shape != other.Shape)
            throw new ArgumentException($"Matrix shapes must match for addition. Got {Shape} and {other.Shape}.");
        
        var result = new double[Data.Length];
        var data1 = Data;
        var data2 = other.Data;

        if (data1.Length < 100_000) ProcessChunk(0, data1.Length, data1, data2, result);
        else
        {
            Parallel.ForEach(
                System.Collections.Concurrent.Partitioner.Create(0, data1.Length),
                (range) =>
                {
                    ProcessChunk(0, data1.Length, data1, data2, result);
                }
            );
        }

        return new Matrix(Shape, result);

        static void ProcessChunk(int start, int end, double[] a, double[] b, double[] res)
        {
            var vectorSize = Vector<double>.Count;
            var i = start;
            
            // Notice the correct <= here!
            for (; i <= end - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(a, i);
                var vb = new Vector<double>(b, i);
                (va * vb).CopyTo(res, i);
            }
            for (; i < end; i++) res[i] = a[i] * b[i];
        }
    }

    /// <summary>
    /// Broadcasts a 1D vector (like biases) across every row of this 2D matrix.
    /// </summary>
    public Matrix BroadcastAdd(Matrix rowVector)
    {
        if (rowVector.Rows != 1)
            throw new ArgumentException("The broadcasted matrix must be a 1D row vector (Rows == 1).");
        
        if (this.Cols != rowVector.Cols)
            throw new ArgumentException("The vector must have the same number of columns as the target matrix.");

        var result = new Matrix(Rows, Cols);
        var thisData = this.Data;
        var otherData = rowVector.Data;
        var resData = result.Data;
        var cols = Cols;

        Parallel.For(0, Rows, i =>
        {
            var rowOffset = i * cols;
            var vectorSize = Vector<double>.Count;
            var j = 0;

            for (; j <= cols - vectorSize; j += vectorSize)
            {
                var vA = new Vector<double>(thisData, rowOffset + j);
                var vB = new Vector<double>(otherData, j);
                (vA + vB).CopyTo(resData, rowOffset + j);
            }

            for (; j < cols; j++)
            {
                resData[rowOffset + j] = thisData[rowOffset + j] + otherData[j];
            }
        });

        return result;
    }

    /// <summary>
    /// Returns true if the matrix is singular (determinant is 0) within the given tolerance.
    /// Singular matrices cannot be inverted and have no unique solution for Ax=b.
    /// </summary>
    public bool IsSingular(double tolerance = 1e-14)
    {
        if (!IsSquare) return true;

        return Math.Abs(Determinant()) < tolerance;
    }

    /// <summary>
    /// Projects this vector onto a target vector. 
    /// Geometrically, this calculates the "shadow" of this vector resting on the target.
    /// </summary>
    public Matrix ProjectOnto(Matrix target, double tolerance = 1e-14)
    {
        if (!IsVector || !target.IsVector)
            throw new InvalidOperationException("Projections are only defined for vectors (N x 1 or 1 x N matrices).");
        
        if (Data.Length != target.Data.Length)
            throw new ArgumentException("Vectors must have the same number of dimensions to be projected.");

        var dotVU = DotProduct(target);
        var dotUU = target.DotProduct(target);
        
        if (Math.Abs(dotUU) < tolerance)
            throw new DivideByZeroException("Cannot project onto a zero vector.");
        
        double scalar = dotVU / dotUU;
        
        return target * scalar;
    }
    
    /// <summary>
    /// Calculates the orthogonal (perpendicular) component of this vector relative to a target vector.
    /// </summary>
    public Matrix OrthogonalComponent(Matrix target)
    {
        // Formula: v_perp = v - proj_u(v)
        return this - ProjectOnto(target);
    }

    /// <summary>
    /// Calculates the mean of each column. 
    /// Optimized using Row-Major memory traversal to maximize L1 Cache hits.
    /// </summary>
    public double[] ColumnMeans()
    {
        var means = new double[Cols];

        for (var i = 0; i < Rows; i++)
        {
            var offset = i * Cols;

            for (var j = 0; j < Cols; j++)
            {
                means[j] += Data[offset + j];
            }
        }

        for (var j = 0; j < Cols; j++)
        {
            means[j] /= Rows;
        }

        return means;
    }

    /// <summary>
    /// Subtracts a mean vector from every row in the matrix, shifting the data's center of gravity to the origin (0,0).
    /// If no mean vector is provided, it calculates the column means automatically.
    /// </summary>
    public Matrix MeanCenterColumns(double[]? providedMeans = null)
    {
        var means = providedMeans ?? ColumnMeans();
        var rows = Rows;
        var cols = Cols;
        var data = Data;
        
        if (means.Length != cols)
            throw new ArgumentException("The mean vector must have exactly one element per column.");

        var centered = new Matrix(Shape);

        if (rows * cols > 100_000)
        {
            Parallel.For(0, rows, i =>
            {
                var offset = i * cols;

                for (var j = 0; j < cols; j++)
                {
                    centered.Data[offset + j] = data[offset + j] - means[j];
                }
            });
        }
        else
        {
            for (var i = 0; i < rows; i++)
            {
                var offset = i * cols;
                
                for (var j = 0; j < cols; j++)
                {
                    centered.Data[offset + j] = Data[offset + j] - means[j];
                }
            }
        }

        return centered;
    }
    
    #endregion
}