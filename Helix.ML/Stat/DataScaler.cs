namespace Helix.ML.Stat;

public static class DataScaler
{
    /// <summary>
    /// Performs in-place Z-Score Standardization (Mean = 0, StdDev = 1).
    /// Modifies the provided span directly.
    /// </summary>
    public static void Standardize(Span<double> data)
    {
        if (data.Length < 2) return;
        
        var (mean, _, stdDev) = DescriptiveStats.ComputeSummary(data, asSample: false);

        if (stdDev == 0)
        {
            data.Fill(0.0);
            return;
        }

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (data[i] - mean) / stdDev;
        }
    }

    /// <summary>
    /// Performs Z-Score Standardization and returns a new array.
    /// </summary>
    public static double[] GetStandardized(ReadOnlySpan<double> data)
    {
        double[] result = data.ToArray();
        Standardize(result);
        
        return result;
    }

    /// <summary>
    /// Performs in-place Min-Max scaling to compress data between 0.0 and 1.0.
    /// Modifies the provided span directly.
    /// </summary>
    public static void MinMaxScale(Span<double> data)
    {
        if (data.IsEmpty) return;
        
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (double val in data)
        {
            if (val < min) min = val;
            if (val > max) max = val;
        }
        
        double range = max - min;

        if (range == 0)
        {
            data.Fill(0.0);
            return;
        }

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (data[i] - min) / range;
        }
    }

    /// <summary>
    /// Performs Min-Max scaling and returns a new array.
    /// </summary>
    public static double[] GetMinMaxScaled(ReadOnlySpan<double> data)
    {
        double[] result = data.ToArray();
        MinMaxScale(result);
        return result;
    }
}