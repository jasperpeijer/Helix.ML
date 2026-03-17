using System.Runtime.CompilerServices;

namespace Helix.ML.Stat;

public static class BivariateStats
{
    /// <summary>
    /// Calculates the covariance of 2 variables
    /// </summary>
    /// <param name="x">The first dataset.</param>
    /// <param name="y">The second dataset.</param>
    /// <param name="asSample">True for sample statistics (n-1), false for population (n).</param>
    /// <returns>A double (Covariance).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Covariance(ReadOnlySpan<double> x, ReadOnlySpan<double> y, bool asSample = false)
    {
        return ComputeBivariateStats(x, y, asSample: asSample).Covariance;
    }
    
    /// <summary>
    /// Calculates the pearson correlation coefficient of 2 variables
    /// </summary>
    /// <param name="x">The first dataset.</param>
    /// <param name="y">The second dataset.</param>
    /// <param name="asSample">True for sample statistics (n-1), false for population (n).</param>
    /// <returns>A double (PearsonCorrelation).</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double PearsonCorrelation(ReadOnlySpan<double> x, ReadOnlySpan<double> y, bool asSample = false)
    {
        return ComputeBivariateStats(x, y, asSample: asSample).PearsonCorrelation;
    }
    
    /// <summary>
    /// Calculates Covariance and Pearson Correlation in a single, robust pass.
    /// </summary>
    /// <param name="x">The first dataset.</param>
    /// <param name="y">The second dataset.</param>
    /// <param name="asSample">True for sample statistics (n-1), false for population (n).</param>
    /// <returns>A tuple containing (Covariance, PearsonCorrelation).</returns>
    public static (double Covariance, double PearsonCorrelation) ComputeBivariateStats(ReadOnlySpan<double> x, ReadOnlySpan<double> y,
        bool asSample = false)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Datasets must be exactly the same length to calculate bivariate statistics.");
        }

        if (x.IsEmpty || x.Length < 2)
        {
            return (0.0, 0.0);
        }

        int n = 0;
        double meanX = 0, meanY = 0;
        double m2X = 0, m2Y = 0;
        double coMoment = 0;

        for (int i = 0; i < x.Length; i++)
        {
            n++;
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;

            meanX += dx / n;
            meanY += dy / n;
            
            m2X += dx * (x[i] - meanX);
            m2Y += dy * (y[i] - meanY);
            
            coMoment += dx * (y[i] - meanY);
        }

        double divisor = asSample ? (n - 1) : n;
        double covariance = coMoment / divisor;
        
        double varX = m2X / divisor;
        double varY = m2Y / divisor;
        
        // Pearson Correlation
        double denominator = Math.Sqrt(varX * varY);
        double correlation = denominator == 0 ? 0 : (covariance / denominator);
        
        return (covariance, correlation);
    }

    /// <summary>
    /// Calculates Spearman's Rank Correlation to measure non-linear monotonic relationships.
    /// </summary>
    /// <param name="x">The first dataset.</param>
    /// <param name="y">The second dataset.</param>
    /// <returns>The Spearman correlation coefficient (-1.0 to 1.0).</returns>
    public static double SpearmanCorrelation(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        if (x.Length != y.Length)
        {
            throw new ArgumentException("Datasets must be exactly the same length.");
        }

        double[] ranksX = Ranker.GetRanks(x);
        double[] ranksY = Ranker.GetRanks(y);

        var result = ComputeBivariateStats(ranksX, ranksY, asSample: false);

        return result.PearsonCorrelation;
    }
}