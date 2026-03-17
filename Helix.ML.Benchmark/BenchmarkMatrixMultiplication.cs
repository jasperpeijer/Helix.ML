using System.Diagnostics;
using Helix.ML.LinAlg;

namespace Helix.ML.Benchmark;

public static class BenchmarkMatrixMultiplication
{
    public static void BenchmarkMultiplication()
    {
        int size = 1000;
        Console.WriteLine($"Allocating two {size}x{size} Random Matrices...");
        
        // Using our brand new factory methods!
        var m1 = Matrix.Random(size, size, -10.0, 10.0);
        var m2 = Matrix.Random(size, size, -10.0, 10.0);

        Console.WriteLine("Warming up the JIT Compiler...");
        var warmupEngine = m1 * m2;
        var warmupNaive = NaiveMultiply(new Matrix(10, 10), new Matrix(10, 10));

        var stopwatch = new Stopwatch();

        // --- Benchmark 1: Textbook Naive Math ---
        Console.WriteLine("\nRunning Naive Textbook Matrix Multiplication...");
        stopwatch.Restart();
        // We only run this 3 times because it is painfully slow
        for (int i = 0; i < 3; i++)
        {
            var result = NaiveMultiply(m1, m2);
        }
        stopwatch.Stop();
        double naiveTime = stopwatch.ElapsedMilliseconds / 3.0;
        Console.WriteLine($"Naive Average: {naiveTime:F2} ms");

        // --- Benchmark 2: Helix Engine ---
        Console.WriteLine("\nRunning Helix High-Performance Engine...");
        stopwatch.Restart();
        // We run yours 10 times to get a stable average
        for (int i = 0; i < 10; i++)
        {
            var result = m1 * m2; // Triggers your custom operator
        }
        stopwatch.Stop();
        double engineTime = stopwatch.ElapsedMilliseconds / 10.0;
        Console.WriteLine($"Engine Average: {engineTime:F2} ms");
        
        // --- The Verdict ---
        Console.WriteLine($"\nSpeedup Multiplier: {naiveTime / engineTime:F2}x Faster!");
    }
    
    /// <summary>
    /// The standard textbook implementation of Matrix Multiplication (Row x Column)
    /// </summary>
    private static Matrix NaiveMultiply(Matrix a, Matrix b)
    {
        var result = new Matrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < b.Cols; j++)
            {
                double sum = 0;
                for (int k = 0; k < a.Cols; k++)
                {
                    // Triggers the indexer bounds checks repeatedly and trashes the cache
                    sum += a[i, k] * b[k, j]; 
                }
                result[i, j] = sum;
            }
        }
        return result;
    }
}