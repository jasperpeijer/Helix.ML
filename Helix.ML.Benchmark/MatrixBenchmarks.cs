using BenchmarkDotNet.Attributes;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Helix.ML.Benchmark;

[MemoryDiagnoser]
[RankColumn]
public class MatrixBenchmarks
{
    private Helix.ML.LinAlg.Matrix _helixMatrix;
    private Matrix<double> _mathNetMatrix;
    
    [Params(10, 100, 500)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rand = new Random(42);
        var data = new double[Size, Size];

        for (var i = 0; i < Size; i++)
        {
            for (var j = 0; j < Size; j++)
            {
                data[i, j] = rand.NextDouble();
            }

            data[i, i] += 10.0;
        }
        
        _helixMatrix = new Helix.ML.LinAlg.Matrix(data);
        _mathNetMatrix = DenseMatrix.OfArray(data);
    }

    [Benchmark(Baseline = true)]
    public void MathNet_Inverse()
    {
        var result = _helixMatrix.Inverse();
    }

    [Benchmark]
    public void Helix_Inverse()
    {
        var result = _helixMatrix.Inverse();
    }

    [Benchmark]
    public void MathNet_Determinant()
    {
        var result = _mathNetMatrix.Determinant();
    }

    [Benchmark]
    public void Helix_Determinant()
    {
        var result = _helixMatrix.Determinant();
    }
}