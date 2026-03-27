using System.Numerics;
using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

/// <summary>
/// A high-performance, contiguous-memory representation of a 2D matrix.
/// </summary>
public readonly partial struct Matrix
{
    public int Rows { get; }
    public int Cols { get; }
    public (int Rows, int Cols) Shape => (Rows, Cols);

    internal readonly double[] Data;

    #region Constructors
    
    /// <summary>
    /// Standard constructor using separate row/col arguments.
    /// </summary>
    public Matrix(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
            throw new ArgumentException("Matrix dimensions must be strictly positive.");
        
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
            throw new ArgumentException("Matrix dimensions must be strictly positive.");

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
            throw new ArgumentException("Matrix dimensions must be strictly positive.");

        if (data.Length != rows * cols)
            throw new ArgumentException("Data length does not match the provided matrix dimensions.");
        
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    /// <summary>
    /// Initializes a matrix from an existing flat 1D array, with a tuple passed for the dimensions
    /// </summary>
    public Matrix((int rows, int cols) shape, double[] data) : this(shape.rows, shape.cols, data) {}
    
    #endregion

    #region Matrix Comparison
    
    public static bool operator ==(Matrix left, Matrix right)
    {
        if (left.Shape != right.Shape) return false;

        const double epsilon = 1e-14;

        return !left.Data.Where((t, i) => Math.Abs(t - right.Data[i]) > epsilon).Any();
    }
    
    public static bool operator !=(Matrix left, Matrix right) => !(left == right);

    public override bool Equals(object? obj) => obj is Matrix other && this == other;

    public override int GetHashCode()
    {
        var hash = HashCode.Combine(Rows, Cols);
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
            throw new ArgumentException($"Cannot compare matrices of different shapes. Got {Shape} and {other.Shape}.");

        for (var i = 0; i < Data.Length; i++)
        {
            var a = Data[i];
            var b = other.Data[i];
            var allowedDifference = atol + (rtol * Math.Abs(b));

            if (Math.Abs(a - b) > allowedDifference) return false;
        }

        return true;
    }
    
    #endregion
}