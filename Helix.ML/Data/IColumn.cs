namespace Helix.ML.Data;

/// <summary>
/// A non-generic interface so the DataFrame can hold a collection of mixed-type columns.
/// </summary>
public interface IColumn
{
    string Name { get; }
    int Length { get; }
    Type DataType { get; }

    object GetValue(int index);
}