using System.Runtime.CompilerServices;

namespace Helix.ML.Stat;

public class DescriptiveStats
{
    /// <summary>
    /// Calculates the Arithmetic Mean (Average) of a data set.
    /// Formula: μ = (Σ x_i) / N
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Mean(ReadOnlySpan<double> data)
    {
        if (data.IsEmpty) return 0;

        double sum = 0;
        
        foreach (var value in data)
        {
            sum += value;
        }
        
        return sum / data.Length;
    }

    /// <summary>
    /// Calculates Population Variance.
    /// Formula: σ² = Σ (x_i - μ)² / N
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Variance(ReadOnlySpan<double> data)
    {
        return ComputeSummary(data).Variance;
    }
    
    /// <summary>
    /// Calculates Standard Deviation.
    /// Formula: σ = √Variance
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double StandardDeviation(ReadOnlySpan<double> data)
    {
        return ComputeSummary(data).StdDev;
    }

    /// <summary>
    /// Calculates Mean, Population Variance, and Standard Deviation in a single pass 
    /// using Welford's robust algorithm.
    /// </summary>
    /// <param name="data">The genomic data set.</param>
    /// <param name="asSample">If true, calculates sample variance (n-1). If false, calculates population variance (n).</param>
    /// <returns>A tuple containing (Mean, Variance, StdDev).</returns>
    public static (double Mean, double Variance, double StdDev) ComputeSummary(ReadOnlySpan<double> data, bool asSample = false)
    {
        if (data.IsEmpty) return (0, 0, 0);

        double mean = 0;
        double m2 = 0;
        int n = 0;

        foreach (double x in data)
        {
            n++;
            double delta = x - mean;
            mean += delta / n;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }

        if (n < 2) return (mean, 0, 0);

        double variance = asSample ? (m2 / (n - 1)) : (m2 / n);
        double stdDev = Math.Sqrt(variance);
        
        return (mean, variance, stdDev);
    }
}