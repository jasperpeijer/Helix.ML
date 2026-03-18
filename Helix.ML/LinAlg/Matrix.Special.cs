namespace Helix.ML.LinAlg;

public readonly partial struct Matrix
{
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
    public static Matrix Zeros((int rows, int cols) shape) => new Matrix(shape.rows, shape.cols);

    /// <summary>
    /// Creates an Identity matrix. If the dimensions are unequal, it places 1.0s on the main diagonal 
    /// until the smallest dimension is exhausted.
    /// </summary>
    public static Matrix Identity(int rows, int cols)
    {
        if (rows <= 0 || cols <= 0)
            throw new ArgumentException("Matrix dimensions must be strictly positive.");    

        var identityMatrix = new Matrix(rows, cols);
        
        var minDim = Math.Min(rows, cols);

        for (var i = 0; i < minDim; i++) identityMatrix.Data[(i * cols) + i] = 1.0;

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

        for (var i = 0; i < matrix.Data.Length; i++)
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

        for (var i = 0; i < matrix.Data.Length; i++)
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
        var rows = Rows;
        var cols = Cols;
        var originalData = Data;
        
        var result = new Matrix(cols, rows);

        Parallel.For(0, Rows, i =>
        {
            var originalRowOffset = i * cols;

            for (var j = 0; j < cols; j++)
            {
                var value = originalData[originalRowOffset + j];
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
}