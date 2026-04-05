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
    /// Gets or sets a column by its string name. 
    /// If the column exists, it is overwritten. If it does not exist, it is appended.
    /// </summary>
    public IColumn this[string columnName]
    {
        get
        {
            var col = Columns.FirstOrDefault(c => c.Name == columnName);
            
            return col ?? throw new KeyNotFoundException($"Column '{columnName}' does not exist.");
        }
        set
        {
            if (value.Length != Rows && Cols > 0)
                throw new ArgumentException($"Row count mismatch. DataFrame has {Rows} rows, but new column has {value.Length}.");

            var renamedCol = value.Rename(columnName);
            var existingIndex = Columns.FindIndex(c => c.Name == columnName);

            if (existingIndex >= 0)
            {
                Columns[existingIndex] = renamedCol;
            }
            else
            {
                Columns.Add(renamedCol);
            }
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
    public void Head(int numRows = 5, int? maxCols = 8)
    {
        Print(numRows, maxCols);
    }
    
    /// <summary>
    /// Prints the DataFrame to the console in a beautifully formatted ASCII table.
    /// </summary>
    public void Print(int? numRows = null, int? maxCols = 8, int colWidth = 15, int indexWidth = 12)
    {
        Console.WriteLine(ToString(numRows, maxCols, colWidth, indexWidth));
    }

    public string ToString(int? numRows, int? maxCols = 8, int colWidth = 15, int indexWidth = 12)
    {
        var displayRows = Math.Min(numRows ?? 10, Rows);
        var truncateCols = maxCols.HasValue && Cols > maxCols.Value;
        var displayCols = new List<IColumn>();
        int leftTake = 0;

        if (truncateCols)
        {
            leftTake = maxCols.Value / 2;
            int rightTake = maxCols.Value - leftTake;
            
            displayCols.AddRange(Columns.Take(leftTake));
            displayCols.AddRange(Columns.Skip(Cols - rightTake));
        }
        else
        {
            displayCols.AddRange(Columns);
        }
        
        var sb = new StringBuilder();
        var tableWidth = (displayCols.Count * colWidth) + indexWidth + (truncateCols ? colWidth : 0) + 1;

        sb.AppendLine($"\nDataFrame: {Rows} rows x {Cols} columns");
        sb.AppendLine(new string('-', tableWidth));
        sb.Append("".PadRight(indexWidth));

        for (var j = 0; j < displayCols.Count; j++)
        {
            if (truncateCols && j == leftTake) sb.Append("...".PadRight(colWidth));

            var col = displayCols[j];
            var header = col.Name.Length > colWidth - 2 ? col.Name.Substring(0, colWidth - 5) + "..." : col.Name;
            sb.Append(header.PadRight(colWidth));
        }
        
        sb.AppendLine();
        
        for (var i = 0; i < displayRows; i++)
        {
            var rowName = Indices[i].Length > indexWidth - 2 ? Indices[i].Substring(0, indexWidth - 3) + "..." : Indices[i];
            sb.Append($"{rowName.PadRight(indexWidth)}");
            
            for (var j = 0; j < displayCols.Count; j++)
            {
                if (truncateCols && j == leftTake) sb.Append("...".PadRight(colWidth));
                
                var rawValue = displayCols[j].GetValue(i);

                var valStr = rawValue switch
                {
                    double d => d.ToString("0.####", CultureInfo.InvariantCulture),
                    _ => rawValue?.ToString() ?? ""
                };
                
                if (valStr.Length > colWidth - 2) 
                    valStr = valStr.Substring(0, colWidth - 5) + "...";
                
                sb.Append(valStr.PadRight(colWidth));
            }
            
            sb.AppendLine();
        }
        
        sb.AppendLine(new string('-', tableWidth) + "\n");

        return sb.ToString();
    }

    public override string ToString()
    {
        return ToString(10, 8);
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
    /// Shuffles the DataFrame and splits it into a Training set and a Testing set.
    /// </summary>
    /// <param name="testRatio">The percentage of data to reserve for testing (e.g., 0.2 = 20%).</param>
    /// <param name="seed">An optional random seed for reproducible splits.</param>
    public (DataFrame train, DataFrame test) TrainTestSplit(double testRatio = 0.2, int? seed = null)
    {
        if (testRatio is <= 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(testRatio), "Test ratio must be strictly between 0.0 and 1.0.");

        var testCount = (int)(Rows * testRatio);
        var trainCount = Rows - testCount;
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var shuffledIndices = Enumerable.Range(0, Rows).OrderBy(x => random.Next()).ToArray();
        var trainIndices = shuffledIndices.Take(trainCount).ToArray();
        var testIndices = shuffledIndices.Skip(trainCount).ToArray();
        var trainColumns = new List<IColumn>(Cols);
        var testColumns = new List<IColumn>(Cols);

        foreach (var col in Columns)
        {
            trainColumns.Add(col.GetRows(trainIndices));
            testColumns.Add(col.GetRows(testIndices));
        }
        
        return (new DataFrame(trainColumns), new DataFrame(testColumns));
    }

    /// <summary>
    /// Extracts the specified numeric columns and converts them into a high-performance Matrix.
    /// If no columns are specified, it extracts all numeric columns.
    /// </summary>
    public Matrix ToMatrix(params string[] columnNames)
    {
        var colsToExtract = columnNames.Length > 0
            ? columnNames
            : Columns.OfType<Column<double>>().Select(c => c.Name).ToArray();
        
        var matrixData = new double[Rows * colsToExtract.Length];

        for (var j = 0; j < colsToExtract.Length; j++)
        {
            var colName = colsToExtract[j];
            var col = this[colName];
            
            if (col is not Column<double> doubleCol)
                throw new InvalidOperationException($"Column '{colName}' must be of type double to convert to a Matrix.");

            for (var i = 0; i < Rows; i++)
            {
                matrixData[i * colsToExtract.Length + j] = doubleCol[i];
            }
        }
        
        return new Matrix(Rows, colsToExtract.Length, matrixData);
    }

    /// <summary>
    /// Creates a complete deep copy of the DataFrame and all its underlying data.
    /// Severes all memory references to the original dataset. Mutating this clone is 100% safe.
    /// </summary>
    public DataFrame Clone()
    {
        var clonedColumns = new List<IColumn>(Cols);
        
        clonedColumns.AddRange(Columns.Select(col => col.Clone()));
        
        var clonedIndices = new List<string>(Indices);
        
        return new DataFrame(clonedColumns, clonedIndices);
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
                var timeData = rawData.Select(val => string.IsNullOrWhiteSpace(val) ? (TimeSpan?)null : TimeSpan.Parse(val, CultureInfo.InvariantCulture)).ToArray();
                finalColumns.Add(new Column<TimeSpan?>(colName, timeData));
            }
            else if (rawData.All(val => 
                     {
                         if (string.IsNullOrWhiteSpace(val)) return true;
                         string[] formats = [
                             // 1. Standard ISO 8601 & API Formats
                             "O", "o", "s", "u", "U", // Built-in C# universal formats
                             "yyyy-MM-ddTHH:mm:ss.fffZ", "yyyy-MM-ddTHH:mm:ssZ", "yyyy-MM-ddTHH:mm:ss",
                             "yyyy-MM-ddTHH:mm:ss.fffffffzzz", "yyyy-MM-ddTHH:mm:ss.fffzzz",
    
                             // 2. Standard SQL & Database Formats (Like the NYC Taxi Dataset)
                             "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss.fff", "yyyy-MM-dd HH:mm",
                             "yyyy-MM-dd", 

                             // 3. Slash Formats (US & International)
                             "MM/dd/yyyy HH:mm:ss", "MM/dd/yyyy hh:mm:ss tt", "MM/dd/yyyy",
                             "dd/MM/yyyy HH:mm:ss", "dd/MM/yyyy hh:mm:ss tt", "dd/MM/yyyy",
                             "yyyy/MM/dd HH:mm:ss", "yyyy/MM/dd",

                             // 4. Dash Formats (Alternate regional exports)
                             "MM-dd-yyyy HH:mm:ss", "MM-dd-yyyy hh:mm:ss tt", "MM-dd-yyyy",
                             "dd-MM-yyyy HH:mm:ss", "dd-MM-yyyy hh:mm:ss tt", "dd-MM-yyyy",

                             // 5. Dot Formats (German & European standards)
                             "dd.MM.yyyy HH:mm:ss", "dd.MM.yyyy",

                             // 6. Text-Month Formats (Excel & Human-typed exports)
                             "dd MMM yyyy", "dd MMMM yyyy", "MMM dd, yyyy", "MMMM dd, yyyy",
                             "dd MMM yyyy HH:mm:ss", "MMM dd, yyyy hh:mm:ss tt"
                         ];
                         return DateTime.TryParseExact(val, formats, CultureInfo.InvariantCulture, DateTimeStyles.None, out _);
                     }))
            {
                var dateData = rawData.Select(val => string.IsNullOrWhiteSpace(val) ? (DateTime?)null : DateTime.Parse(val, CultureInfo.InvariantCulture)).ToArray();
                finalColumns.Add(new Column<DateTime?>(colName, dateData));
            }
            else
            {
                finalColumns.Add(new Column<string>(colName, rawData));
            }
        }
        
        return new DataFrame(finalColumns);
    }

    /// <summary>
    /// Universally encodes boolean and string columns into ML-ready binary double columns.
    /// Utilizes multithreading to construct the encoded arrays at high speeds.
    /// </summary>
    public DataFrame Encode(params string[] columnNames)
    {
        var newColumns = new List<IColumn>(Cols);
        var colsToEncode = new HashSet<string>(columnNames);
        
        if (columnNames.Length == 0)
        {
            foreach (var col in Columns)
            {
                if (col is not Column<double> _ or Column<int>)
                    colsToEncode.Add(col.Name);
            }
        }

        foreach (var col in Columns)
        {
            if (!colsToEncode.Contains(col.Name))
            {
                newColumns.Add(col);
                continue;
            }

            if (col is Column<bool?> boolCol)
            {
                var doubleData = new double[Rows];

                Parallel.For(0, Rows, i =>
                {
                    doubleData[i] = boolCol[i] == true ? 1.0 : 0.0;
                });
                
                newColumns.Add(new Column<double>(col.Name, doubleData));
            }
            else if (col is Column<string> stringCol)
            {
                var uniqueVals = new HashSet<string>();

                for (var i = 0; i < Rows; i++)
                {
                    if (!string.IsNullOrEmpty(stringCol[i]))
                        uniqueVals.Add(stringCol[i]);
                }

                foreach (var category in uniqueVals)
                {
                    var binaryData = new double[Rows];

                    Parallel.For(0, Rows, i =>
                    {
                        binaryData[i] = stringCol[i] == category ? 1.0 : 0.0;
                    });
                    
                    newColumns.Add(new Column<double>($"{col.Name}_{category}", binaryData));
                }
            }
            else if (col is Column<DateTime?> dateCol)
            {
                var years = new double[Rows];
                var months = new double[Rows];
                var days = new double[Rows];
                var dayOfWeeks = new double[Rows];
                var hours = new double[Rows];
                var minutes = new double[Rows];
                var seconds = new double[Rows];
                var milliseconds = new double[Rows];

                Parallel.For(0, Rows, (int i) =>
                {
                    var dt = dateCol[i];

                    if (dt.HasValue)
                    {
                        years[i] = dt.Value.Year;
                        months[i] = dt.Value.Month;
                        days[i] = dt.Value.Day;
                        hours[i] = dt.Value.Hour;
                        minutes[i] = dt.Value.Minute;
                        seconds[i] = dt.Value.Second;
                        milliseconds[i] = dt.Value.Millisecond;
                        dayOfWeeks[i] = (double)dt.Value.DayOfWeek;
                    }
                    else
                    {
                        years[i] = double.NaN;
                        months[i] = double.NaN;
                        days[i] = double.NaN;
                        hours[i] = double.NaN;
                        minutes[i] = double.NaN;
                        seconds[i] = double.NaN;
                        milliseconds[i] = double.NaN;
                        dayOfWeeks[i] = double.NaN;
                    }
                });
                
                newColumns.Add(new Column<double>($"{col.Name}_Year", years));
                newColumns.Add(new Column<double>($"{col.Name}_Month", months));
                newColumns.Add(new Column<double>($"{col.Name}_Day", days));
                newColumns.Add(new Column<double>($"{col.Name}_Hour", hours));
                newColumns.Add(new Column<double>($"{col.Name}_Minute", minutes));
                newColumns.Add(new Column<double>($"{col.Name}_Second", seconds));
                newColumns.Add(new Column<double>($"{col.Name}_Millisecond", milliseconds));
                newColumns.Add(new Column<double>($"{col.Name}_DayOfWeek", dayOfWeeks));
            }
            else if (col is Column<TimeSpan?> timeCol)
            {
                var totalSeconds = new double[Rows];

                Parallel.For(0, Rows, (int i) =>
                {
                    var ts = timeCol[i];
                    totalSeconds[i] = ts?.TotalSeconds ?? double.NaN;
                });
                
                newColumns.Add(new Column<double>($"{col.Name}_TotalSeconds", totalSeconds));
            }
            else
            {
                throw new InvalidOperationException($"Cannot encode column '{col.Name}'. Type {col.DataType.Name} is not supported.");
            }
        }
        
        return new DataFrame(newColumns, Indices);
    }

    /// <summary>
    /// Horizontally combines two DataFrames [Left | Right].
    /// Useful for adding new feature columns to an existing dataset.
    /// </summary>
    public DataFrame Augment(DataFrame right)
    {
        if (Rows != right.Rows)
            throw new ArgumentException($"Row counts must match. Left has {Rows}, Right has {right.Rows}.");
        
        var newColumns = new List<IColumn>(Columns);
        var existingNames = new HashSet<string>(ColumnNames);

        foreach (var col in right.Columns)
        {
            if (existingNames.Contains(col.Name))
                throw new InvalidOperationException($"Column name collision: '{col.Name}' already exists.");
            
            newColumns.Add(col);
        }
        
        return new DataFrame(newColumns, Indices);
    }

    /// <summary>
    /// Syntactic sugar for Augmenting two DataFrames [A | B].
    /// </summary>
    public static DataFrame operator |(DataFrame left, DataFrame right)
        => left.Augment(right);

    /// <summary>
    /// Vertically combines two DataFrames, stacking them on top of each other.
    /// Useful for combining batches of data.
    /// </summary>
    public DataFrame Concatenate(DataFrame bottom)
    {
        if (this.Cols != bottom.Cols)
            throw new ArgumentException("DataFrames must have the exact same number of columns to be concatenated vertically.");
        
        var newColumns = new List<IColumn>(Cols);

        for (var j = 0; j < Cols; j++)
        {
            var topCol = Columns[j];
            var bottomCol = bottom.Columns[j];
            
            if (topCol.Name != bottomCol.Name || topCol.DataType != bottomCol.DataType)
                throw new ArgumentException($"Column mismatch at index {j}. Expected '{topCol.Name}' ({topCol.DataType.Name}), got '{bottomCol.Name}' ({bottomCol.DataType.Name}).");
            
            newColumns.Add(topCol.Concat(bottomCol));
        }

        var newIndices = new List<string>(this.Indices);
        newIndices.AddRange(bottom.Indices);
        
        return new DataFrame(newColumns, newIndices);
    }

    /// <summary>
    /// Syntactic sugar for Concatenating two DataFrames vertically.
    /// </summary>
    public static DataFrame operator &(DataFrame top, DataFrame bottom)
        => top.Concatenate(bottom);

    /// <summary>
    /// Safely extracts a strongly-typed column for high-speed indexing.
    /// </summary>
    public Column<T> GetColumn<T>(string columnName)
    {
        var col = this[columnName];
        
        if (col is not Column<T> typedCol)
            throw new InvalidOperationException($"Column '{columnName}' is not of type {typeof(T).Name}.");
        
        return typedCol;
    }

    /// <summary>
    /// The ultimate filter. Allows infinite logical and mathematical expressions across mixed data types.
    /// Operates at native hardware speeds by completely avoiding dictionary lookups and boxing.
    /// </summary>
    public DataFrame Filter(Func<int, bool> predicate)
    {
        var passedIndices = new List<int>(Rows / 2);

        for (var i = 0; i < Rows; i++)
        {
            if (predicate(i)) passedIndices.Add(i);
        }

        var finalIndices = passedIndices.ToArray();
        var newColumns = new List<IColumn>(Cols);

        foreach (var col in Columns) newColumns.Add(col.GetRows(finalIndices));

        var newIndices = finalIndices.Select(x => Indices[x]).ToList();
        
        return new DataFrame(newColumns, newIndices);
    }

    /// <summary>
    /// Filters the DataFrame using a Vectorized Boolean Mask.
    /// Operates at extreme speeds by evaluating a pre-calculated true/false array.
    /// </summary>
    public DataFrame Filter(Column<bool> mask)
    {
        if (mask.Length != Rows)
            throw new ArgumentException($"Mask length ({mask.Length}) must exactly match DataFrame rows ({Rows}).");

        var passedIndices = new List<int>(Rows / 2);

        for (var i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                passedIndices.Add(i);
            }
        }
        
        var finalIndices = passedIndices.ToArray();
        var newColumns = new List<IColumn>(Cols);
        
        foreach (var col in Columns) newColumns.Add(col.GetRows(finalIndices));
        
        var newIndices = finalIndices.Select(x => Indices[x]).ToList();
        
        return new DataFrame(newColumns, newIndices);
    }
    
    /// <summary>
    /// Converts a 2D Matrix back into a DataFrame.
    /// </summary>
    public static DataFrame FromMatrix(Matrix matrix, string[]? columnNames = null)
    {
        var newCols = new List<IColumn>(matrix.Cols);

        for (var j = 0; j < matrix.Cols; j++)
        {
            var name = columnNames != null && j < columnNames.Length
                ? columnNames[j]
                : $"Column{j}";

            var colData = new double[matrix.Rows];

            for (var i = 0; i < matrix.Rows; i++)
            {
                colData[i] = matrix[i, j];
            }
            
            newCols.Add(new Column<double>(name, colData));
        }
        
        return new DataFrame(newCols);
    }
}