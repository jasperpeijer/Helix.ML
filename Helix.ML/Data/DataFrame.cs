using System.Buffers;
using System.Globalization;
using System.Text;
using Helix.ML.LinAlg;
using Helix.ML.Stat;

namespace Helix.ML.Data;

/// <summary>
/// A high-level data structure for manipulating tabular data, wrapping the high-performance Matrix engine.
/// </summary>
public class DataFrame
{
    public List<IColumn> Columns { get; private set; }

    public List<string> Indices { get; private set; }

    public int Rows => Columns.Count > 0 ? Columns[0].Length : 0;
    public int Cols => Columns.Count;
    public List<string> ColumnNames => Columns.Select(c => c.Name).ToList();

    public DataFrame(IEnumerable<IColumn> columns, IEnumerable<string>? indices = null)
    {
        Columns = columns.ToList();
        
        if (Columns.Count == 0)
            throw new ArgumentException("DataFrame must contain at least one column.");

        var expectedLength = Columns[0].Length;
        
        foreach (var col in Columns)
        {
            if (col.Length != expectedLength)
                throw new ArgumentException($"Column '{col.Name}' has {col.Length} rows, but expected {expectedLength}.");
        }
        
        Indices = indices?.ToList() ?? Enumerable.Range(0, expectedLength).Select(i => i.ToString()).ToList();
        
        if (Indices.Count != expectedLength)
            throw new ArgumentException($"Matrix has {expectedLength} rows, but {Indices.Count} row names were provided.");
    }

    /// <summary>
    /// Extracts a single column by name, returning it as an Nx1 Matrix ready for linear algebra.
    /// </summary>
    public IColumn this[string columnName]
    {
        get
        {
            var col = Columns.FirstOrDefault(c => c.Name == columnName);
            
            return col ?? throw new KeyNotFoundException($"Column '{columnName}' does not exist.");
        }
    }

    /// <summary>
    /// Slices the DataFrame, returning a brand new DataFrame containing only the requested columns.
    /// </summary>
    public DataFrame Select(params string[] columnNames)
    {
        var newColumns = new List<IColumn>();

        foreach (var columnName in columnNames)
        {
            var col = Columns.FirstOrDefault(c => c.Name == columnName);
            
            if (col == null)
                throw new KeyNotFoundException($"Column '{columnName}' does not exist.");
            
            newColumns.Add(col);
        }
        
        return new DataFrame(newColumns, Indices);
    }

    /// <summary>
    /// Prints the first N rows of the DataFrame to the console in a beautifully formatted table.
    /// </summary>
    public void Head(int numRows = 5)
    {
        Print(numRows);
    }
    
    /// <summary>
    /// Prints the DataFrame to the console in a beautifully formatted ASCII table.
    /// </summary>
    public void Print(int? numRows = null)
    {
        Console.WriteLine(ToString(numRows));
    }

    public string ToString(int? numRows)
    {
        var displayRows = Math.Min(numRows ?? 10, Rows);
        var colWidth = 15;
        var indexWidth = 12;
        var sb = new StringBuilder();

        sb.AppendLine($"\nDataFrame: {Rows} rows x {Cols} columns");
        sb.AppendLine(new string('-', (Cols * colWidth) + indexWidth + Cols + 1));
        sb.Append("".PadRight(indexWidth));
        
        foreach (var col in Columns)
        {
            var header = col.Name.Length > colWidth - 2 ? col.Name.Substring(0, colWidth - 5) + "..." : col.Name;
            sb.Append(header.PadRight(colWidth));
        }
        
        sb.AppendLine();
        
        for (var i = 0; i < displayRows; i++)
        {
            var rowName = Indices[i].Length > indexWidth - 2 ? Indices[i].Substring(0, indexWidth - 3) + "..." : Indices[i];
            sb.Append($"{rowName.PadRight(indexWidth)}");
            
            for (var j = 0; j < Cols; j++)
            {
                var rawValue = Columns[j].GetValue(i);
                var valStr = "";

                if (rawValue is double d) 
                    valStr = d.ToString("0.####", CultureInfo.InvariantCulture);
                else valStr = rawValue.ToString() ?? "";
                
                if (valStr.Length > colWidth - 2) 
                    valStr = valStr.Substring(0, colWidth - 5) + "...";
                
                sb.Append(valStr.PadRight(colWidth));
            }
            
            sb.AppendLine();
        }
        
        sb.AppendLine(new string('-', (Cols * colWidth) + indexWidth + Cols + 2) + "\n");

        return sb.ToString();
    }

    public override string ToString()
    {
        return ToString(10);
    }
    
    /// <summary>
    /// Generates descriptive statistics (Count, Mean, Std, Min, Max) for all numeric columns.
    /// Non-numeric columns are automatically ignored.
    /// </summary>
    public DataFrame Describe()
    {
        var numericCols = Columns.OfType<Column<double>>().ToList();
        
        if (numericCols.Count == 0)
            throw new InvalidOperationException("DataFrame contains no numeric columns to describe.");
        
        string[] statNames = ["Count", "Mean", "Std", "Min", "Max"];
        var resultColumns = new IColumn[numericCols.Count];

        Parallel.For(0, numericCols.Count, j =>
        {
            var col = numericCols[j];
            var colBuffer = ArrayPool<double>.Shared.Rent(Rows);

            try
            {
                var min = double.MaxValue;
                var max = double.MinValue;
                var validCount = 0;

                for (var i = 0; i < Rows; i++)
                {
                    var val = col[i];

                    if (double.IsNaN(val)) continue;
                    
                    colBuffer[validCount] = val;
                    validCount++;
                    
                    if (val < min) min = val;
                    if (val > max) max = val;
                }

                if (validCount == 0)
                {
                    resultColumns[j] =
                        new Column<double>(col.Name, [0, double.NaN, double.NaN, double.NaN, double.NaN]);
                    return;
                }

                var span = colBuffer.AsSpan(0, validCount);
                var summary = DescriptiveStats.ComputeSummary(span, asSample: true);
                var statValues = new double[] { validCount, summary.Mean, summary.StdDev, min, max };
                resultColumns[j] = new Column<double>(col.Name, statValues);
            }
            finally
            {
                ArrayPool<double>.Shared.Return(colBuffer);
            }
        });
        
        return new DataFrame(resultColumns, statNames);
    }

    /// <summary>
    /// Returns a summary DataFrame containing column data types and non-null counts.
    /// </summary>
    public DataFrame Info()
    {
        var types = new string[Cols];
        var nonNulls = new double[Cols];
        var missing = new double[Cols];

        for (var j = 0; j < Cols; j++)
        {
            var col = Columns[j];
                
            var underlyingType = Nullable.GetUnderlyingType(col.DataType);
            types[j] = underlyingType != null ? underlyingType.Name + "?" : col.DataType.Name;

            var validCount = 0;

            for (var i = 0; i < Rows; i++)
            {
                var val = col.GetValue(i);

                var isValid = val switch
                {
                    double d => !double.IsNaN(d),
                    string s => !string.IsNullOrWhiteSpace(s),
                    DateTime dt => dt != DateTime.MinValue,
                    TimeSpan ts => ts != TimeSpan.Zero,
                    _ => val != null
                };

                if (isValid) validCount++;
            }

            nonNulls[j] = validCount;
            missing[j] = Rows - validCount;
        }

        var typeColumn = new Column<string>("DataType", types);
        var validColumn = new Column<double>("NonNull_Count", nonNulls);
        var missingColumn = new Column<double>("Missing_Count", missing);

        return new DataFrame([typeColumn, validColumn, missingColumn], ColumnNames);
    }

    /// <summary>
    /// Reads a CSV file, infers column types (double, bool, string), and returns a new DataFrame.
    /// </summary>
    public static DataFrame LoadCsv(string filePath, bool hasHeader = true, char separator = ',')
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Could not find the file: {filePath}");

        var lines = File.ReadAllLines(filePath);
        
        if (lines.Length == 0)
            throw new InvalidDataException("The CSV file is empty.");
        
        var startIndex = hasHeader ? 1 : 0;

        var headers = hasHeader ? lines[0].Split(separator).Select(h => h.Trim()).ToList() :
            Enumerable.Range(0, lines[0].Split(separator).Length).Select(i => $"Column{i}").ToList();

        var numCols = headers.Count;
        var numRows = lines.Length - startIndex;
        var rawColumns = new List<string>[numCols];

        for (var j = 0; j < numCols; j++)
            rawColumns[j] = new List<string>(numRows);

        for (var i = startIndex; i < lines.Length; i++)
        {
            var cells = lines[i].Split(separator);

            for (var j = 0; j < numCols; j++)
            {
                rawColumns[j].Add(j < cells.Length ? cells[j].Trim() : string.Empty);
            }
        }

        var finalColumns = new List<IColumn>();

        for (var j = 0; j < numCols; j++)
        {
            var rawData = rawColumns[j];
            var colName = headers[j];

            if (rawData.All(val =>
                    string.IsNullOrWhiteSpace(val) ||
                    double.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out _)))
            {
                var doubleData = rawData.Select(val =>
                        string.IsNullOrWhiteSpace(val) ? 
                            double.NaN : 
                            double.Parse(val, CultureInfo.InvariantCulture)
                ).ToArray();
                finalColumns.Add(new Column<double>(colName, doubleData));
            }
            else if (rawData.All(val => string.IsNullOrWhiteSpace(val) || bool.TryParse(val, out _)))
            {
                var boolData = rawData.Select(val => string.IsNullOrWhiteSpace(val) ? (bool?)null : bool.Parse(val)).ToArray();
                finalColumns.Add(new Column<bool?>(colName, boolData));
            }
            else if (rawData.All(val => string.IsNullOrWhiteSpace(val) || TimeSpan.TryParse(val, CultureInfo.InvariantCulture, out _)))
            {
                var timeData = rawData.Select(val => string.IsNullOrWhiteSpace(val) ? TimeSpan.Zero : TimeSpan.Parse(val, CultureInfo.InvariantCulture)).ToArray();
                finalColumns.Add(new Column<TimeSpan>(colName, timeData));
            }
            else if (rawData.All(val => 
                     {
                         if (string.IsNullOrWhiteSpace(val)) return true;
                         string[] formats = ["yyyy-MM-dd", "MM/dd/yyyy", "dd-MM-yyyy", "yyyy/MM/dd HH:mm:ss", "O"];
                         return DateTime.TryParseExact(val, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out _);
                     }))
            {
                var dateData = rawData.Select(val => string.IsNullOrWhiteSpace(val) ? DateTime.MinValue : DateTime.Parse(val, CultureInfo.InvariantCulture)).ToArray();
                finalColumns.Add(new Column<DateTime>(colName, dateData));
            }
            else
            {
                finalColumns.Add(new Column<string>(colName, rawData));
            }
        }
        
        return new DataFrame(finalColumns);
    }
}