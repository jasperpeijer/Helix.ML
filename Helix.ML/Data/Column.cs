namespace Helix.ML.Data;

/// <summary>
/// A strongly-typed column representing a single feature in a DataFrame.
/// </summary>
public class Column<T>(string name, IEnumerable<T> data) : IColumn
{
    public string Name { get; } = name;
    public T[] Data { get; private set; } = data.ToArray();
    public int Length => Data.Length;
    public Type DataType => typeof(T);

    public object GetValue(int index)
    {
        if (index < 0 || index >= Length)
            throw new IndexOutOfRangeException($"Index {index} is out of bounds for column '{Name}'.");

        return Data[index];
    }

    public IColumn GetRows(int[] indices)
    {
        var newData = new T[indices.Length];

        for (var i = 0; i < indices.Length; i++)
        {
            newData[i] = Data[indices[i]];
        }

        return new Column<T>(Name, newData);
    }

    public IColumn Concat(IColumn bottom)
    {
        if (bottom is not Column<T> bottomCol)
            throw new ArgumentException(
                $"Cannot concatenate column of type {bottom.DataType.Name} onto {DataType.Name}.");

        var newData = new T[Length + bottom.Length];

        Array.Copy(Data, 0, newData, 0, Length);
        Array.Copy(bottomCol.Data, 0, newData, Length, bottomCol.Length);

        return new Column<T>(Name, newData);
    }

    /// <summary>
    /// Creates a clone of the column with a new name.
    /// </summary>
    public IColumn Rename(string newName)
    {
        return new Column<T>(newName, Data);
    }

    /// <summary>
    /// Creates a deep copy of the column, allocating a brand new array in memory.
    /// Severes any reference sharing with the original column.
    /// </summary>
    public IColumn Clone()
    {
        var newArray = new T[Length];
        Array.Copy(Data, newArray, Length);

        return new Column<T>(Name, newArray);
    }

    /// <summary>
    /// Subtracts another DateTime column from this one, returning a new TimeSpan column.
    /// </summary>
    public Column<TimeSpan?> SubtractDates(Column<DateTime?> other)
    {
        if (typeof(T) != typeof(DateTime?)) throw new InvalidOperationException("This method requires a DateTime? column.");

        var (thisData, otherData) = (Data as DateTime?[], other.Data as DateTime?[]);
        var res = new TimeSpan?[Length];
        Parallel.For(0, Length, i =>
        {
            if (thisData[i].HasValue && otherData[i].HasValue)
            {
                res[i] = thisData[i] - otherData[i];
            }
            else
            {
                res[i] = null;
            }
        });
        
        return new Column<TimeSpan?>($"{Name}_{other.Name}_Duration", res);
    }

    /// <summary>
    /// Projects each element of the column into a new form using a vectorized lambda function.
    /// Allows seamless data type transformations (e.g., DateTime to TimeSpan, string to int).
    /// </summary>
    public Column<TU> Map<TU>(Func<T, TU> mapper)
    {
        var res = new TU[Length];

        Parallel.For(0, Length, i =>
        {
            res[i] = mapper(Data[i]);
        });
        
        return new Column<TU>($"{Name}_Mapped", res);
    }

    public T this[int index]
    {
        get => Data[index];
        set => Data[index] = value;
    }

    // =========================================================================
    // 1. MATHEMATICAL OPERATORS (+, -, *, /)
    // =========================================================================

    public static Column<T> operator +(Column<T> left, Column<T> right)
    {
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var res = new T[left.Length];

        if (typeof(T) == typeof(double))
        {
            var (l, r) = (left.Data as double[], right.Data as double[]);
            var resDouble = new double[left.Length];
            Parallel.For(0, left.Length, i => resDouble[i] = l[i] + r[i]);
        }
        else if (typeof(T) == typeof(string))
        {
            var (l, r) = (left.Data as string[], right.Data as string[]);
            var resString = new string[left.Length];
            Parallel.For(0, left.Length, i => resString[i] = l[i] + r[i]);
        }
        else
        {
            throw new InvalidOperationException($"Operator '+' is not supported for type {typeof(T).Name}.");
        }

        return new Column<T>($"{left.Name}_add_{right.Name}", res);
    }

    public static Column<double> operator +(Column<T> col, double scalar)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = data[i] + scalar);

        return new Column<double>($"{col.Name}_add_{scalar}", res);
    }

    public static Column<double> operator +(double scalar, Column<T> col) => col + scalar;

    public static Column<double> operator -(Column<T> left, Column<T> right)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double columns.");
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var (lData, rData) = (left.Data as double[], right.Data as double[]);
        var res = new double[left.Length];
        Parallel.For(0, left.Length, i => res[i] = lData[i] - rData[i]);

        return new Column<double>($"{left.Name}_sub_{right.Name}", res);
    }

    public static Column<double> operator -(Column<T> col, double scalar) => col + (-scalar);

    public static Column<double> operator -(double scalar, Column<T> col)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = scalar - data[i]);

        return new Column<double>($"{scalar}_sub_{col.Name}", res);
    }

    public static Column<double> operator *(Column<T> left, Column<T> right)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double columns.");
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var (lData, rData) = (left.Data as double[], right.Data as double[]);
        var res = new double[left.Length];
        Parallel.For(0, left.Length, i => res[i] = lData[i] * rData[i]);

        return new Column<double>($"{left.Name}_mult_{right.Name}", res);
    }

    public static Column<double> operator *(Column<T> col, double scalar)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = data[i] * scalar);

        return new Column<double>($"{col.Name}_mult_{scalar}", res);
    }

    public static Column<double> operator *(double scalar, Column<T> col) => col * scalar;

    public static Column<double> operator /(Column<T> left, Column<T> right)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double columns.");
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var (lData, rData) = (left.Data as double[], right.Data as double[]);
        var res = new double[left.Length];
        Parallel.For(0, left.Length, i => res[i] = lData[i] / rData[i]);

        return new Column<double>($"{left.Name}_div_{right.Name}", res);
    }

    public static Column<double> operator /(Column<T> col, double scalar)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = data[i] / scalar);

        return new Column<double>($"{col.Name}_div_{scalar}", res);
    }

    public static Column<double> operator /(double scalar, Column<T> col)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = scalar / data[i]);

        return new Column<double>($"{scalar}_div_{col.Name}", res);
    }

    public static Column<double> operator -(Column<T> col)
    {
        if (typeof(T) != typeof(double)) throw new InvalidOperationException("Math requires double column.");
        
        var data = col.Data as double[];
        var res = new double[col.Length];
        Parallel.For(0, col.Length, i => res[i] = -data[i]);
        
        return new Column<double>($"{col.Name}_neg", res);
    }

    // =========================================================================
    // 2. BOOLEAN MASKS & COMPARISONS (<, >, <=, >=, ==, !=)
    // =========================================================================
    public static Column<bool> operator >(Column<T> left, Column<T> right)
    {
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var res = new bool[left.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, left.Length, i => res[i] = comparer.Compare(left.Data[i], right.Data[i]) > 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator >(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(col.Data[i], value) > 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator >(T value, Column<T> col)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(value, col.Data[i]) > 0);
        
        return new Column<bool>("Mask", res);
    }

    public static Column<bool> operator >=(Column<T> left, Column<T> right)
    {
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var res = new bool[left.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, left.Length, i => res[i] = comparer.Compare(left.Data[i], right.Data[i]) >= 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator >=(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(col.Data[i], value) >= 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator >=(T value, Column<T> col)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(value, col.Data[i]) >= 0);
        
        return new Column<bool>("Mask", res);
    }

    public static Column<bool> operator <(Column<T> left, Column<T> right)
    {
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var res = new bool[left.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, left.Length, i => res[i] = comparer.Compare(left.Data[i], right.Data[i]) < 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator <(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(col.Data[i], value) < 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator <(T value, Column<T> col)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(value, col.Data[i]) < 0);
        
        return new Column<bool>("Mask", res);
    }

    public static Column<bool> operator <=(Column<T> left, Column<T> right)
    {
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        
        var res = new bool[left.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, left.Length, i => res[i] = comparer.Compare(left.Data[i], right.Data[i]) <= 0);
        
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator <=(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(col.Data[i], value) <= 0);
        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator <=(T value, Column<T> col)
    {
        var res = new bool[col.Length];
        var comparer = Comparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Compare(value, col.Data[i]) <= 0);
        
        return new Column<bool>("Mask", res);
    }

    public static Column<bool> operator ==(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = EqualityComparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = comparer.Equals(col.Data[i], value));
        return new Column<bool>("Mask", res);
    }

    public static Column<bool> operator !=(Column<T> col, T value)
    {
        var res = new bool[col.Length];
        var comparer = EqualityComparer<T>.Default;
        Parallel.For(0, col.Length, i => res[i] = !comparer.Equals(col.Data[i], value));
        return new Column<bool>("Mask", res);
    }

    // =========================================================================
    // 3. LOGICAL OPERATORS (&, |)
    // =========================================================================
    public static Column<bool> operator &(Column<T> left, Column<T> right)
    {
        if (typeof(T) != typeof(bool)) throw new InvalidOperationException("Logical AND requires boolean columns.");
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        var (l, r) = (left.Data as bool[], right.Data as bool[]);
        var res = new bool[left.Length];
        Parallel.For(0, left.Length, i => res[i] = l[i] && r[i]);

        return new Column<bool>("Mask", res);
    }
    
    public static Column<bool> operator |(Column<T> left, Column<T> right)
    {
        if (typeof(T) != typeof(bool)) throw new InvalidOperationException("Logical OR requires boolean columns.");
        if (left.Length != right.Length) throw new ArgumentException("Columns must match in length.");
        var (l, r) = (left.Data as bool[], right.Data as bool[]);
        var res = new bool[left.Length];
        Parallel.For(0, left.Length, i => res[i] = l[i] || r[i]);

        return new Column<bool>("Mask", res);
    }

}