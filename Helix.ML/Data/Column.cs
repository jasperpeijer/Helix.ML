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

    public T this[int index]
    {
        get => Data[index];
        set => Data[index] = value;
    }
}