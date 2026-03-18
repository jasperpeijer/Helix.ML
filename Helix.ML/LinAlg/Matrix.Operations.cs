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
        double[] result = new double[m.Data.Length];
        int i = 0;
        int vectorSize = Vector<double>.Count;
        var data = m.Data;
        
        // Broadcast the single scalar value into a full hardware vector
        var vScalar = new Vector<double>(scalar);

        if (data.Length < 100_000) {
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
        }
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length), range =>
            {
                int i = range.Item1;
                int vectorSize = Vector<double>.Count;
                var vScalar = new Vector<double>(scalar);
                
                for (; i <= range.Item2 - vectorSize; i += vectorSize)
                {
                    var va = new Vector<double>(data, i);
                    (va * vScalar).CopyTo(result, i);
                }
                for (; i < range.Item2; i++) result[i] = data[i] * scalar;
            });
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
        var data = m.Data;

        if (data.Length < 100_000) {
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
        }
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data.Length), range =>
            {
                int i = range.Item1;
                int vectorSize = Vector<double>.Count;
                var vScalar = new Vector<double>(scalar);

                for (; i <= range.Item2 - vectorSize; i += vectorSize)
                {
                    var vA = new Vector<double>(data, i);
                    (vA + vScalar).CopyTo(result, i);
                }
                for (; i < range.Item2; i++) result[i] = data[i] + scalar;
            });
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
        var vectorSize = Vector<double>.Count;
        var data1 = m1.Data;
        var data2 = m2.Data;

        if (data1.Length < 100_000)
        {
            for (; i < m1.Data.Length - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(m1.Data, i);
                var vb = new Vector<double>(m2.Data, i);
                var vResult = va + vb;

                vResult.CopyTo(result, i);
            }

            for (; i < m1.Data.Length; i++) result[i] = m1.Data[i] + m2.Data[i];
        }
        else
        {
            Parallel.ForEach(
                System.Collections.Concurrent.Partitioner.Create(0, data1.Length),
                (range) =>
                {
                    int i = range.Item1;
                    int vectorSize = Vector<double>.Count;

                    for (; i <= range.Item2 - vectorSize; i += vectorSize)
                    {
                        var va = new Vector<double>(data1, i);
                        var vb = new Vector<double>(data2, i);
                        (va + vb).CopyTo(result, i);
                    }

                    for (; i < range.Item2; i++) result[i] = data1[i] + data2[i];
                }
            );
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
        var vectorSize = Vector<double>.Count;
        var data1 = m1.Data;
        var data2 = m2.Data;
        
        if (data1.Length < 100_000) {
            for (; i < m1.Data.Length - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(m1.Data, i);
                var vb = new Vector<double>(m2.Data, i);
                var vResult = va - vb;

                vResult.CopyTo(result, i);
            }

            for (; i < m1.Data.Length; i++)
            {
                result[i] = m1.Data[i] - m2.Data[i];
            }
        }
        else
        {
            Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, data1.Length), range =>
            {
                int i = range.Item1;
                int vectorSize = Vector<double>.Count;
                for (; i <= range.Item2 - vectorSize; i += vectorSize)
                {
                    var va = new Vector<double>(data1, i);
                    var vb = new Vector<double>(data2, i);
                    (va - vb).CopyTo(result, i);
                }
                for (; i < range.Item2; i++) result[i] = data1[i] - data2[i];
            });
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

    /// <summary>
    /// Calculates the flat, element-wise Dot Product of two matrices/vectors.
    /// </summary>
    public double DotProduct(Matrix other)
    {
        if (this.Data.Length != other.Data.Length)
        {
            throw new ArgumentException("Matrices must have the same total number of elements to calculate a flat dot product.");
        }
        
        var thisData = this.Data;
        var otherData = other.Data;

        if (Data.Length < 100_000) {
            var sum = 0.0;
            int vectorSize = Vector<double>.Count;
            int i = 0;

            for (; i < thisData.Length - vectorSize; i += vectorSize)
            {
                var va = new Vector<double>(thisData, i);
                var vb = new Vector<double>(otherData, i);
                sum += Vector.Dot(va, vb);
            }

            for (; i < thisData.Length; i++)
            {
                sum += thisData[i] * otherData[i];
            }

            return sum;
        }

        double globalSum = 0.0;
        object lockObj = new object();

        Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(0, thisData.Length),
            () => 0.0,
            (range, loopState, localSum) =>
            {
                int vectorSize = Vector<double>.Count;
                int i = 0;

                for (; i < thisData.Length - vectorSize; i += vectorSize)
                {
                    var va = new Vector<double>(thisData, i);
                    var vb = new Vector<double>(otherData, i);
                    localSum += Vector.Dot(va, vb);
                }

                for (; i < thisData.Length; i++)
                {
                    localSum += thisData[i] * otherData[i];
                }

                return localSum;
            },
            (localSum) => { lock (lockObj) globalSum += localSum; }
        );

        return globalSum;
    }
    
    #endregion
}