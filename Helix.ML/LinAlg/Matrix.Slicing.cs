using System.Runtime.CompilerServices;

namespace Helix.ML.LinAlg;

public partial struct Matrix
{
    /// <summary>
    /// Extracts specific rows by their exact indices.
    /// Utilizes native memory copying and multithreading for massive datasets.
    /// </summary>
    public Matrix ExtractRows(params int[] rowIndices)
    {
        if (rowIndices == null || rowIndices.Length == 0)
            throw new ArgumentException("Must provide at least one row index.");

        var result = new Matrix(rowIndices.Length, Cols);
        var cols = Cols;
        var srcData = Data;
        var dstData = result.Data;
        
        foreach (var i in rowIndices)
        {
            if (i < 0 || i >= Rows)
                throw new IndexOutOfRangeException($"Row index {i} is out of bounds.");
        }

        if (rowIndices.Length * cols < 100_000)
        {
            for (var i = 0; i < rowIndices.Length; i++)
            {
                Array.Copy(srcData, rowIndices[i] * cols, dstData, i * cols, cols);
            }
        }
        else
        {
            Parallel.For(0, rowIndices.Length, i =>
            {
                Array.Copy(srcData, rowIndices[i] * cols, dstData, i * cols, cols);
            });
        }
        
        return result;
    }

    /// <summary>
    /// Extracts specific columns by their exact indices (e.g., keeping only features 0, 2, and 5).
    /// </summary>
    public Matrix ExtractColumns(params int[] colIndices)
    {
        if (colIndices == null || colIndices.Length == 0)
            throw new ArgumentException("Must provide at least one column index.");

        var result = new Matrix(Rows, colIndices.Length);
        var rows = Rows;
        var cols = Cols;
        var resCols = colIndices.Length;
        var srcData = Data;
        var dstData = result.Data;

        foreach (var j in colIndices)
        {
            if (j < 0 || j >= Cols)
                throw new IndexOutOfRangeException($"Column index {j} is out of bounds.");
        }

        if (rows * resCols < 100_000)
        {
            for (var i = 0; i < rows; i++)
            {
                var srcOffset = i * cols;
                var dstOffset = i * resCols;

                for (var j = 0; j < resCols; j++)
                {
                    dstData[dstOffset + j] = srcData[srcOffset + colIndices[j]];
                }
            }
        }
        else
        {
            Parallel.For(0, rows, i =>
            {
                var srcOffset = i * cols;
                var dstOffset = i * resCols;

                for (var j = 0; j < resCols; j++)
                {
                    dstData[dstOffset + j] = srcData[srcOffset + colIndices[j]];
                }
            });
        }

        return result;
    }

    /// <summary>
    /// Filters the matrix, keeping only the rows that satisfy a specific condition.
    /// Evaluates a user-provided predicate function against each row.
    /// </summary>
    public Matrix FilterRows(Func<double[], bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);

        var keptIndices = new List<int>();
        var rowBuffer = new double[Cols];

        for (var i = 0; i < Rows; i++)
        {
            Array.Copy(Data, i * Cols, rowBuffer, 0, Cols);

            if (predicate(rowBuffer))
            {
                keptIndices.Add(i);
            }
        }
        
        if (keptIndices.Count == 0)
            throw new InvalidOperationException("Filter resulted in an empty matrix.");

        return ExtractRows([..keptIndices]);
    }
    
    #region Matrix Indexing & Slicing
    
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

            for (var i = 0; i < rowLength; i++)
            {
                var srcIndex = ((rowOffset + i) * Cols) + colOffset;
                var dstIndex = i * colLength;
                
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
                var srcIndex = i * value.Cols;
                var dstIndex = ((rowOffset + i) * Cols) + colOffset;
                
                Array.Copy(value.Data, srcIndex, this.Data, dstIndex, colLength);
            }
        }
    }
    
    #endregion
}