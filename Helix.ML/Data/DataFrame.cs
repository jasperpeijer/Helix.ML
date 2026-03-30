using Helix.ML.LinAlg;

namespace Helix.ML.Data;

/// <summary>
/// A high-level data structure for manipulating tabular data, wrapping the high-performance Matrix engine.
/// </summary>
public class DataFrame
{
    /// <summary>
    /// The underlying hardware-accelerated math engine.
    /// </summary>
    public Matrix CoreMatrix { get; private set; }
    
    /// <summary>
    /// The ordered list of column names.
    /// </summary>
    public List<string> Columns { get; private set; }

    public int Rows => CoreMatrix.Rows;
    public int Cols => CoreMatrix.Cols;

    public DataFrame(Matrix data, IEnumerable<string> columns)
    {
        var colList = columns.ToList();
        
        if (data.Cols != colList.Count)
            throw new ArgumentException($"Matrix has {data.Cols} columns, but {colList.Count} names were provided.");
        
        CoreMatrix = data;
        Columns = colList;
    }

    /// <summary>
    /// Extracts a single column by name, returning it as an Nx1 Matrix ready for linear algebra.
    /// </summary>
    public Matrix this[string columnName]
    {
        get
        {
            var colIndex = Columns.IndexOf(columnName);
            
            if (colIndex == -1)
                throw new KeyNotFoundException($"Column '{columnName}' does not exist.");
            
            return CoreMatrix.ExtractColumns(colIndex + 1);
        }
    }

    /// <summary>
    /// Slices the DataFrame, returning a brand new DataFrame containing only the requested columns.
    /// </summary>
    public DataFrame Select(params string[] columnNames)
    {
        var indices = new List<int>();
        
        foreach (var columnName in columnNames)
        {
            var index = Columns.IndexOf(columnName);
            
            if (index == -1)
                throw new KeyNotFoundException($"Column '{columnName}' does not exist.");
            
            indices.Add(index);
        }

        var newMatrix = CoreMatrix.ExtractColumns([..indices]);
        
        return new DataFrame(newMatrix, columnNames);
    }

    /// <summary>
    /// Prints the first N rows of the DataFrame to the console in a beautifully formatted table.
    /// </summary>
    public void Head(int numRows = 5)
    {
        var displayRows = Math.Min(Rows, numRows);
        var colWidth = 15;
        
        Console.WriteLine($"\nDataFrame: {Rows} rows x {Cols} columns");
        Console.WriteLine(new string('-', (Cols * colWidth) + Cols + 1));

        foreach (var col in Columns)
        {
            var header = col.Length > colWidth - 2 ? col.Substring(0, colWidth - 5) + "..." : col;
            Console.Write(header.PadRight(colWidth));
        }
        
        Console.WriteLine();

        for (var i = 0; i < displayRows; i++)
        {

            for (var j = 0; j < Cols; j++)
            {
                var val = CoreMatrix[i, j].ToString("0.####");
                Console.Write(val.PadRight(colWidth));
            }
            
            Console.WriteLine();
        }
        
        Console.WriteLine(new string('-', (Cols * colWidth) + Cols + 1) + "\n");
    }
}