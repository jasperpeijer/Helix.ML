using Helix.ML.LinAlg;
using Xunit.Abstractions;

namespace Helix.ML.Tests;

public class MatrixTests
{
    private readonly ITestOutputHelper _testOutputHelper;

    public MatrixTests(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    [Fact]
    public void Constructor_ValidDimensions_CreatesCorrectShape()
    {
        // Act
        var m = new Matrix(3, 4);

        // Assert
        Assert.Equal((3, 4), m.Shape);
        Assert.Equal(3, m.Shape.Rows);
        Assert.Equal(4, m.Shape.Cols);
    }
    
    [Theory]
    [InlineData(0, 5)]
    [InlineData(5, 0)]
    [InlineData(-1, 5)]
    [InlineData(5, -1)]
    public void Constructor_InvalidDimensions_ThrowsArgumentException(int rows, int cols)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Matrix(rows, cols));
    }
    
    [Theory]
    [InlineData(0, 5)]
    [InlineData(5, 0)]
    [InlineData(-1, 5)]
    [InlineData(5, -1)]
    public void Constructor_WithTuple_InvalidDimensions_ThrowsArgumentException(int rows, int cols)
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Matrix((rows, cols)));
    }
    
    [Fact]
    public void Constructor_2DArray_FlattensCorrectly()
    {
        // Arrange
        double[,] data = 
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        };

        // Act
        var m = new Matrix(data);

        // Assert
        Assert.Equal((2, 3), m.Shape);
        Assert.Equal(2, m.Shape.Rows);
        Assert.Equal(3, m.Shape.Cols);
        Assert.Equal(1.0, m[0, 0]);
        Assert.Equal(2.0, m[0, 1]);
        Assert.Equal(3.0, m[0, 2]);
        Assert.Equal(4.0, m[1, 0]);
        Assert.Equal(5.0, m[1, 1]);
        Assert.Equal(6.0, m[1, 2]);
    }
    
    [Fact]
    public void Constructor_1DArray_MismatchedLength_ThrowsArgumentException()
    {
        // Arrange
        double[] data = [1.0, 2.0, 3.0, 4.0]; // Length 4

        // Act & Assert
        // Trying to pack 4 elements into a 2x3 (6 element) matrix should fail
        Assert.Throws<ArgumentException>(() => new Matrix(2, 3, data));
    }
    
    [Fact]
    public void Indexer_SetsAndGetsCorrectly_UsingRowMajorOrder()
    {
        // Arrange
        var m = new Matrix(2, 2);

        // Act
        m[0, 0] = 10.0;
        m[0, 1] = 20.0;
        m[1, 0] = 30.0;
        m[1, 1] = 40.0;

        // Assert
        Assert.Equal(10.0, m[0, 0]);
        Assert.Equal(20.0, m[0, 1]);
        Assert.Equal(30.0, m[1, 0]);
        Assert.Equal(40.0, m[1, 1]);
    }
    
    [Fact]
    public void OperatorSubtract_SameShape_SubtractsElementWise()
    {
        // Arrange
        var a = new Matrix(new double[,] { { 5.0, 10.0 }, { 15.0, 20.0 } });
        var b = new Matrix(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var result = a - b;

        // Assert
        Assert.Equal(4.0, result[0, 0]);
        Assert.Equal(8.0, result[0, 1]);
        Assert.Equal(12.0, result[1, 0]);
        Assert.Equal(16.0, result[1, 1]);
    }
    
    [Fact]
    public void OperatorSubtract_DifferentShapes_ThrowsArgumentException()
    {
        // Arrange
        var a = new Matrix(2, 2);
        var b = new Matrix(3, 3);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a - b);
    }
    
    [Fact]
    public void OperatorMultiply_ByDoubleScalar_ScalesAllElements()
    {
        // Arrange
        var m = new Matrix(new double[,] { { 1.0, -2.0 }, { 3.0, 4.0 } });

        // Act
        var result1 = m * 2.5; 
        var result2 = 3.0 * m; // Testing the commutative overload

        // Assert
        Assert.Equal(2.5, result1[0, 0]);
        Assert.Equal(-5.0, result1[0, 1]);
        Assert.Equal(7.5, result1[1, 0]);
        Assert.Equal(10, result1[1, 1]);
        
        Assert.Equal(3.0, result2[0, 0]);
        Assert.Equal(-6.0, result2[0, 1]);
        Assert.Equal(9.0, result2[1, 0]);
        Assert.Equal(12.0, result2[1, 1]);
    }
    
    [Fact]
    public void OperatorMultiply_ByInteger_ImplicitlyCastsToDouble()
    {
        // Arrange
        var m = new Matrix(new double[,] { { 1.0, 2.0 } });

        // Act
        // We pass an int (5), proving the compiler handles the conversion
        var result = m * 5; 

        // Assert
        Assert.Equal(5.0, result[0, 0]);
        Assert.Equal(10.0, result[0, 1]);
    }
    
    [Fact]
    public void Zeros_ValidDimensions_CreatesMatrixOfAllZeros()
    {
        // Arrange & Act
        var m = Matrix.Zeros(2, 3);

        // Assert
        Assert.Equal((2, 3), m.Shape);
        Assert.Equal(0.0, m[0, 0]);
        Assert.Equal(0.0, m[0, 1]);
        Assert.Equal(0.0, m[0, 2]);
        Assert.Equal(0.0, m[1, 0]);
        Assert.Equal(0.0, m[1, 1]);
        Assert.Equal(0.0, m[1, 2]);
    }
    
    [Fact]
    public void Identity_SquareDimensions_CreatesStandardIdentity()
    {
        // Arrange & Act
        var m = Matrix.Identity(3);

        // Assert
        Assert.Equal((3, 3), m.Shape);
        
        // Main diagonal should be 1.0, everything else should be 0
        Assert.Equal(1.0, m[0, 0]);
        Assert.Equal(0.0, m[0, 1]);
        Assert.Equal(0.0, m[0, 2]);
        Assert.Equal(1.0, m[1, 1]);
        Assert.Equal(0.0, m[1, 0]);
        Assert.Equal(0.0, m[1, 2]);
        Assert.Equal(1.0, m[2, 2]);
        Assert.Equal(0.0, m[2, 0]);
        Assert.Equal(0.0, m[2, 1]);
    }
    
    [Fact]
    public void Identity_RectangularTall_StopsDiagonalSafely()
    {
        // Arrange: 4 rows, 2 columns (Tall)
        // Act
        var m = Matrix.Identity(4, 2);

        // Assert
        Assert.Equal((4, 2), m.Shape);
        Assert.Equal(1.0, m[0, 0]);
        Assert.Equal(1.0, m[1, 1]);
        
        // Row 2 and 3 have no matching columns, so they should be pure zeros
        Assert.Equal(0.0, m[0, 1]); 
        Assert.Equal(0.0, m[1, 0]); 
        Assert.Equal(0.0, m[2, 0]); 
        Assert.Equal(0.0, m[2, 1]); 
        Assert.Equal(0.0, m[3, 0]); 
        Assert.Equal(0.0, m[3, 1]); 
    }
    
    [Fact]
    public void Identity_RectangularWide_StopsDiagonalSafely()
    {
        // Arrange: 2 rows, 4 columns (Wide)
        // Act
        var m = Matrix.Identity(2, 4);

        // Assert
        Assert.Equal((2, 4), m.Shape);
        Assert.Equal(1.0, m[0, 0]);
        Assert.Equal(1.0, m[1, 1]);
        
        // Columns 2 and 3 have no matching rows, so they should be pure zeros
        Assert.Equal(0.0, m[0, 1]);
        Assert.Equal(0.0, m[0, 2]);
        Assert.Equal(0.0, m[0, 3]);
        Assert.Equal(0.0, m[1, 0]);
        Assert.Equal(0.0, m[1, 2]);
        Assert.Equal(0.0, m[1, 3]);
    }
    
    [Fact]
    public void Random_WithTupleShapeAndRange_CreatesValuesInBounds()
    {
        // Arrange & Act
        // Using the new tuple overloads!
        var m = Matrix.Random((10, 10), (-5.0, 5.0));

        // Assert
        Assert.Equal((10, 10), m.Shape);
        
        // Loop through the flat array to ensure no number escaped the bounds
        foreach (double val in m.Data)
        {
            Assert.InRange(val, -5.0, 5.0);
        }
    }
    
    [Fact]
    public void RandomInt_WithTupleShapeAndRange_CreatesIntegersInBounds()
    {
        // Arrange & Act
        var m = Matrix.RandomInt((5, 5), (-10, 10));

        // Assert
        Assert.Equal((5, 5), m.Shape);
        
        foreach (double val in m.Data)
        {
            // Prove it stayed in bounds
            Assert.InRange(val, -10, 10);
            
            // Prove it is actually a whole number, even though it is stored as a double
            Assert.Equal(Math.Truncate(val), val);
        }
    }
    
    [Fact]
    public void OperatorMultiply_MatrixDotProduct_CalculatesCorrectly()
    {
        // Arrange: 2x3 matrix
        var a = new Matrix(new double[,] 
        { 
            { 1.0, 2.0, 3.0 }, 
            { 4.0, 5.0, 6.0 } 
        });

        // Arrange: 3x2 matrix
        var b = new Matrix(new double[,] 
        { 
            { 7.0, 8.0 }, 
            { 9.0, 10.0 }, 
            { 11.0, 12.0 } 
        });

        // Act: (2x3) * (3x2) = (2x2)
        var result = a * b;

        // Assert
        Assert.Equal((2, 2), result.Shape);
        
        // Row 0 * Col 0: (1*7) + (2*9) + (3*11) = 58
        Assert.Equal(58.0, result[0, 0]);
        // Row 0 * Col 1: (1*8) + (2*10) + (3*12) = 64
        Assert.Equal(64.0, result[0, 1]);
        // Row 1 * Col 0: (4*7) + (5*9) + (6*11) = 139
        Assert.Equal(139.0, result[1, 0]);
        // Row 1 * Col 1: (4*8) + (5*10) + (6*12) = 154
        Assert.Equal(154.0, result[1, 1]);
    }
    
    [Fact]
    public void OperatorMultiply_InnerDimensionsMismatch_ThrowsArgumentException()
    {
        // Arrange
        var a = new Matrix(2, 3); // Inner dimension is 3
        var b = new Matrix(4, 5); // Inner dimension is 4

        // Act & Assert
        Assert.Throws<ArgumentException>(() => a * b);
    }
    
    [Fact]
    public void OperatorMultiply_WithIdentityMatrix_LeavesMatrixUnchanged()
    {
        // Arrange: Using our new random factory and identity factory!
        var m = Matrix.RandomInt(3, 3, -10, 10);
        var i = Matrix.Identity(3);

        // Act
        var result = m * i;

        // Assert
        for (int row = 0; row < m.Rows; row++)
        {
            for (int col = 0; col < m.Cols; col++)
            {
                // The output matrix should be identical to the input matrix
                Assert.Equal(m[row, col], result[row, col]);
            }
        }
    }
    
    [Fact]
    public void Transpose_RectangularMatrix_FlipsDimensionsAndData()
    {
        // Arrange: 2x3 Matrix
        var m = new Matrix(new double[,] {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act
        var mT = m.Transpose();

        // Assert
        Assert.Equal(3, mT.Rows);
        Assert.Equal(2, mT.Cols);
        
        // Check specific values to ensure the shuffle worked
        Assert.Equal(1, mT[0, 0]); // Was [0,0]
        Assert.Equal(4, mT[0, 1]); // Was [1,0]
        Assert.Equal(2, mT[1, 0]); // Was [0,1]
        Assert.Equal(5, mT[1, 1]); // Was [0,1]
        Assert.Equal(3, mT[2, 0]); // Was [1,2]
        Assert.Equal(6, mT[2, 1]); // Was [1,2]
    }
    
    [Fact]
    public void Transpose_DoubleTranspose_ReturnsOriginal()
    {
        // Arrange
        // Using our tuple randomizer!
        var a = Matrix.Random((10, 5), (-10.0, 10.0));

        // Act: (A^T)^T
        var result = a.Transpose().Transpose();

        // Assert
        Assert.Equal(a.Shape, result.Shape);
        for (int i = 0; i < a.Data.Length; i++)
        {
            Assert.Equal(a.Data[i], result.Data[i]);
        }
    }
    
    [Fact]
    public void PropertyT_ReturnsTransposedMatrix()
    {
        // Arrange: 2x3 Matrix
        var m = new Matrix(new double[,] {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act
        var mT = m.T;

        // Assert
        Assert.Equal(3, mT.Rows);
        Assert.Equal(2, mT.Cols);
        
        // Check specific values to ensure the shuffle worked
        Assert.Equal(1, mT[0, 0]); // Was [0,0]
        Assert.Equal(4, mT[0, 1]); // Was [1,0]
        Assert.Equal(2, mT[1, 0]); // Was [0,1]
        Assert.Equal(5, mT[1, 1]); // Was [0,1]
        Assert.Equal(3, mT[2, 0]); // Was [1,2]
        Assert.Equal(6, mT[2, 1]); // Was [1,2]
    }
    
    [Fact]
    public void Trace_SquareMatrix_ReturnsSumOfMainDiagonal()
    {
        // Arrange
        var m = new Matrix(new double[,] {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        double trace = m.Trace();

        // Assert
        // Diagonal is 1.0 + 5.0 + 9.0 = 15.0
        Assert.Equal(15.0, trace); 
    }
    
    [Fact]
    public void Trace_NonSquareMatrix_ThrowsInvalidOperationException()
    {
        // Arrange
        var m = Matrix.Random(2, 5); // Rectangular

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => m.Trace());
    }
    
    [Fact]
    public void EqualityOperator_SameValuesWithTinyMathError_ReturnsTrue()
    {
        // Arrange
        var m1 = new Matrix(new double[,] { { 0.1 + 0.2 } }); 
        var m2 = new Matrix(new double[,] { { 0.3 } });

        // Act & Assert
        // In pure binary, 0.1 + 0.2 = 0.30000000000000004. 
        // If our epsilon works, this will correctly evaluate to true!
        Assert.True(m1 == m2); 
    }
    
    [Fact]
    public void EqualityOperator_DifferentValues_ReturnsFalse()
    {
        // Arrange
        var m1 = Matrix.Zeros(2, 2);
        var m2 = Matrix.Zeros(2, 2);
        m2[1, 1] = 0.0001; // Tiny difference, but way larger than 1e-14

        // Act & Assert
        Assert.False(m1 == m2);
        Assert.True(m1 != m2);
    }
    
    [Fact]
    public void Trace_RectangularMatrixWithFlag_ReturnsSumOfAvailableDiagonal()
    {
        // Arrange: 2x3 Matrix
        var m = new Matrix(new double[,] {
            { 10.0, 20.0, 30.0 },
            { 40.0, 50.0, 60.0 }
        });

        // Act
        // We explicitly allow it. It should sum 10.0 and 50.0 and stop.
        double trace = m.Trace(allowRectangular: true);

        // Assert
        Assert.Equal(60.0, trace);
    }
    
    [Fact]
    public void IsCloseTo_WithTinyAndMassiveNumbers_CorrectlyEvaluatesTolerance()
    {
        // Arrange
        // We have a microscopic number, and a massive number
        var m1 = new Matrix(new double[,] { { 1e-10, 1000000.0 } });
        
        // We add a tiny bit of floating-point noise to both
        var m2 = new Matrix(new double[,] { { 1.1e-10, 1000000.001 } });
        
        // Act & Assert
        // Standard == will fail because 0.001 is much larger than our strict 1e-14 epsilon
        Assert.False(m1 == m2);
        
        // But IsCloseTo will pass because 0.001 is a microscopic difference *relative* to 1,000,000
        Assert.True(m1.IsCloseTo(m2));
    }
    
    [Fact]
    public void Indexer_WithRanges_ExtractsCorrectSubMatrix()
    {
        // Arrange: A 4x4 matrix
        // [  1,  2,  3,  4 ]
        // [  5,  6,  7,  8 ]
        // [  9, 10, 11, 12 ]
        // [ 13, 14, 15, 16 ]
        var m = new Matrix(new double[,] {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 },
            { 13, 14, 15, 16 }
        });

        // Act 1: Standard Range (Rows 1 to 2, Cols 1 to 2)
        // Note: The end of a Range is EXCLUSIVE in C#
        var sub1 = m[1..3, 1..3];

        // Act 2: Using the 'from end' operator ^
        // ^3 means "3 from the end" (index 1), ^1 means "1 from the end" (index 3)
        var sub2 = m[^3..^1, 1..3];

        // Act 3: Open-ended ranges
        // Start at 1 and go to 3, start at 1 and go to 3
        var sub3 = m[1..3, 1..^1];

        // Assert: All three should extract the exact same 2x2 center block
        // [  6,  7 ]
        // [ 10, 11 ]
        
        Assert.Equal((2, 2), sub1.Shape);
        Assert.Equal(6, sub1[0, 0]);
        Assert.Equal(7, sub1[0, 1]);
        Assert.Equal(10, sub1[1, 0]);
        Assert.Equal(11, sub1[1, 1]);

        // Prove the alternate syntaxes worked identically
        Assert.True(sub1 == sub2);
        Assert.True(sub1 == sub3);
    }
    
    [Fact]
    public void Augment_TwoMatrices_CombinesThemHorizontally()
    {
        // Arrange
        var left = new Matrix(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        
        var right = new Matrix(new double[,] {
            { 5 },
            { 6 }
        });

        // Act
        var augmented = left.Augment(right);

        // Assert
        Assert.Equal(2, augmented.Rows);
        Assert.Equal(3, augmented.Cols);
        
        Assert.Equal(1, augmented[0, 0]);
        Assert.Equal(2, augmented[0, 1]);
        Assert.Equal(5, augmented[0, 2]);
        Assert.Equal(3, augmented[1, 0]);
        Assert.Equal(4, augmented[1, 1]);
        Assert.Equal(6, augmented[1, 2]);
    }
    
    [Fact]
    public void Concatenate_TwoMatrices_CombinesThemVertically()
    {
        // Arrange
        var top = new Matrix(new double[,] { { 1, 2, 3 } });
        var bottom = new Matrix(new double[,] { { 4, 5, 6 } });

        // Act
        var stacked = top.Concatenate(bottom);

        // Assert
        Assert.Equal(2, stacked.Rows);
        Assert.Equal(3, stacked.Cols);
        
        Assert.Equal(1, stacked[0, 0]); // First element of the bottom row
        Assert.Equal(2, stacked[0, 1]); // First element of the bottom row
        Assert.Equal(3, stacked[0, 2]); // First element of the bottom row
        Assert.Equal(4, stacked[1, 0]); // First element of the bottom row
        Assert.Equal(5, stacked[1, 1]); // First element of the bottom row
        Assert.Equal(6, stacked[1, 2]); // First element of the bottom row
    }
    
    [Fact]
    public void OperatorBitwiseOr_ActsAsAugment()
    {
        // Arrange
        var left = new Matrix(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });
        
        var right = new Matrix(new double[,] {
            { 5 },
            { 6 }
        });

        // Act
        var augmented = left | right;

        // Assert
        Assert.Equal(2, augmented.Rows);
        Assert.Equal(3, augmented.Cols);
        
        Assert.Equal(1, augmented[0, 0]);
        Assert.Equal(2, augmented[0, 1]);
        Assert.Equal(5, augmented[0, 2]);
        Assert.Equal(3, augmented[1, 0]);
        Assert.Equal(4, augmented[1, 1]);
        Assert.Equal(6, augmented[1, 2]);
    }
    
    [Fact]
    public void IncrementOperator_AddsOneToEveryElement()
    {
        // Arrange
        var m = Matrix.Zeros(2, 2);

        // Act
        m++;

        // Assert
        foreach (double val in m.Data)
        {
            Assert.Equal(1.0, val);
        }
    }
    
    [Fact]
    public void DecrementOperator_SubtractsOneFromEveryElement()
    {
        // Arrange
        var m = Matrix.Zeros(2, 2);

        // Act
        m--;

        // Assert
        foreach (double val in m.Data)
        {
            Assert.Equal(-1.0, val);
        }
    }
    
    [Fact]
    public void OperatorBitwiseAnd_ActsAsConcatenateWithBlockCopy()
    {
        // Arrange
        var top = new Matrix(new double[,] { { 1, 2 } });
        var bottom = new Matrix(new double[,] { { 3, 4 } });

        // Act
        var result = top & bottom; // Uses the block-copy & operator!

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Cols);
        
        Assert.Equal(1, result[0, 0]); 
        Assert.Equal(2, result[0, 1]);
        Assert.Equal(3, result[1, 0]); 
        Assert.Equal(4, result[1, 1]);
    }
    
    [Fact]
    public void UnaryNegation_FlipsAllSigns()
    {
        // Arrange
        var m = new Matrix(new double[,] { { 1, -2 }, { 0, 3 } });

        // Act
        var result = -m;

        // Assert
        Assert.Equal(-1, result[0, 0]);
        Assert.Equal(2, result[0, 1]);
        Assert.Equal(0, result[1, 0]);
        Assert.Equal(-3, result[1, 1]);
    }
    
    [Fact]
    public void OperatorAdd_SameShape_AddsElementWise()
    {
        var a = new Matrix(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = new Matrix(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });

        var result = a + b;

        Assert.Equal(6.0, result[0, 0]);
        Assert.Equal(8.0, result[0, 1]);
        Assert.Equal(10.0, result[1, 0]);
        Assert.Equal(12.0, result[1, 1]);
    }
    
    [Fact]
    public void OperatorDivide_ByScalar_DividesAllElements()
    {
        var m = new Matrix(new double[,] { { 10.0, 20.0 }, { -5.0, 0.0 } });

        var result = m / 2.0;

        Assert.Equal(5.0, result[0, 0]);
        Assert.Equal(10.0, result[0, 1]);
        Assert.Equal(-2.5, result[1, 0]);
        Assert.Equal(0.0, result[1, 1]);
    }
    
    [Fact]
    public void Exceptions_VariousShapeMismatches_ThrowArgumentExceptions()
    {
        var m2x2 = Matrix.Zeros(2, 2);
        var m2x3 = Matrix.Zeros(2, 3);
        var m3x3 = Matrix.Zeros(3, 3);
        var m3x2 = Matrix.Zeros(3, 2);

        // Addition & Subtraction Mismatches
        Assert.Throws<ArgumentException>(() => m2x2 + m3x3);
        
        // Augment (Horizontal Stack) Mismatches - Requires same row count
        Assert.Throws<ArgumentException>(() => m2x2.Augment(m3x3));
        
        // Concatenate (Vertical Stack) Mismatches - Requires same column count
        Assert.Throws<ArgumentException>(() => m2x3.Concatenate(m3x2));
        
        // Range Setter Mismatches - Trying to put a 3x3 into a 2x2 slice
        Assert.Throws<ArgumentException>(() => m2x2[0..2, 0..2] = m3x3);
        
        // IsCloseTo Mismatches
        Assert.Throws<ArgumentException>(() => m2x2.IsCloseTo(m3x3));
    }
    
    [Fact]
    public void Constructor_1DArray_InjectsCorrectly()
    {
        double[] data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        // 2 rows, 3 cols requires exactly 6 elements
        var m = new Matrix(2, 3, data); 
        
        Assert.Equal(3.0, m[0, 2]);
        Assert.Equal(6.0, m[1, 2]);
    }
    
    [Fact]
    public void Ones_ValidDimensions_CreatesMatrixOfAllOnes()
    {
        // Arrange & Act
        var m = Matrix.Ones(2, 3);

        // Assert
        Assert.Equal((2, 3), m.Shape);
        
        foreach (double val in m.Data)
        {
            Assert.Equal(1.0, val);
        }
    }
    
    [Fact]
    public void OperatorAdd_ByScalar_AddsToAllElements()
    {
        // Arrange
        var m = new Matrix(new double[,] { { 1.0, -2.0 }, { 0.0, 4.0 } });

        // Act
        var result1 = m + 5.0;
        var result2 = 5.0 + m; // Proves the commutative overload works

        // Assert
        Assert.Equal(6.0, result1[0, 0]);
        Assert.Equal(3.0, result1[0, 1]);
        Assert.Equal(5.0, result1[1, 0]);
        Assert.Equal(9.0, result1[1, 1]);
        
        // Ensure result2 matches result1
        Assert.True(result1 == result2);
    }
    
    [Fact]
    public void OperatorSubtract_ByScalar_SubtractsFromAllElements()
    {
        // Arrange
        var m = new Matrix(new double[,] { { 1.0, -2.0 }, { 0.0, 4.0 } });

        // Act
        var result = m - 5.0;

        // Assert
        Assert.Equal(-4.0, result[0, 0]);
        Assert.Equal(-7.0, result[0, 1]);
        Assert.Equal(-5.0, result[1, 0]);
        Assert.Equal(-1.0, result[1, 1]);
        
        // Ensure result2 matches result1
    }
    
    [Fact]
    public void Determinant_2x2Matrix_CalculatesCorrectly()
    {
        // Arrange
        var m = new Matrix(new double[,] {
            { 3, 8 },
            { 4, 6 }
        });

        // Act
        double det = m.Determinant();

        // Assert
        // (3 * 6) - (8 * 4) = 18 - 32 = -14
        Assert.Equal(-14.0, det);
    }
    
    [Fact]
    public void Determinant_3x3Matrix_CalculatesCorrectly()
    {
        // Arrange
        var m = new Matrix(new double[,] {
            { 6, 1, 1 },
            { 4, -2, 5 },
            { 2, 8, 7 }
        });

        // Act
        double det = m.Determinant();

        // Assert
        // A standard 3x3 test case: det should be -306
        Assert.Equal(-306.0, det);
    }
    
    [Fact]
    public void Determinant_NonSquareMatrix_ThrowsInvalidOperationException()
    {
        var m = Matrix.Zeros(2, 3);
        Assert.Throws<InvalidOperationException>(() => m.Determinant());
    }
    
    [Fact]
    public void IsTriangular_UpperAndLower_EvaluatesCorrectly()
    {
        var upper = new Matrix(new double[,] {
            { 1, 2, 3 },
            { 0, 4, 5 },
            { 0, 0, 6 }
        });

        var lower = new Matrix(new double[,] {
            { 1, 0, 0 },
            { 2, 4, 0 },
            { 3, 5, 6 }
        });

        Assert.True(upper.IsUpperTriangular());
        Assert.False(upper.IsLowerTriangular());

        Assert.True(lower.IsLowerTriangular());
        Assert.False(lower.IsUpperTriangular());

        // Prove IsDiagonal works
        var diag = Matrix.Identity(3);
        Assert.True(diag.IsDiagonal());
    }
    
    [Fact]
    public void Determinant_TriangularMatrix_UsesDiagonalFastPath()
    {
        // Arrange
        var upper = new Matrix(new double[,] {
            { 2, 8, 9, 4 },
            { 0, 3, 7, 5 },
            { 0, 0, 4, 2 },
            { 0, 0, 0, 5 }
        });

        // Act
        double det = upper.Determinant();

        // Assert
        // The determinant must be 2 * 3 * 4 * 5 = 120
        Assert.Equal(120.0, det);
    }
    
    [Fact]
    public void IsSquare_EvaluatesCorrectly()
    {
        var square = Matrix.Zeros(3, 3);
        var rectangular = Matrix.Zeros(2, 3);

        Assert.True(square.IsSquare);
        Assert.False(rectangular.IsSquare);
    }
    
    [Fact]
    public void IsSymmetric_EvaluatesCorrectly()
    {
        // A perfectly symmetric matrix (mirrored across the main diagonal)
        var symmetric = new Matrix(new double[,] {
            { 1, 7, 3 },
            { 7, 4, -5 },
            { 3, -5, 6 }
        });

        // A slightly altered matrix that breaks symmetry
        var nonSymmetric = new Matrix(new double[,] {
            { 1, 7, 3 },
            { 8, 4, -5 }, // 8 does not match 7
            { 3, -5, 6 }
        });

        Assert.True(symmetric.IsSymmetric());
        Assert.False(nonSymmetric.IsSymmetric());
    }
    
    [Fact]
    public void IsOrthogonal_EvaluatesCorrectly()
    {
        // The Identity matrix is perfectly orthogonal
        var identity = Matrix.Identity(3);
        
        // A 90-degree 2D rotation matrix is also orthogonal
        var rotation = new Matrix(new double[,] {
            { 0, -1 },
            { 1,  0 }
        });

        var standard = new Matrix(new double[,] {
            { 1, 2 },
            { 3, 4 }
        });

        Assert.True(identity.IsOrthogonal());
        Assert.True(rotation.IsOrthogonal());
        Assert.False(standard.IsOrthogonal());
    }
    
    [Fact]
    public void IsAntiSymmetric_EvaluatesCorrectly()
    {
        // A standard 3x3 skew-symmetric matrix. 
        // Notice the diagonal is all 0s, and the mirrored elements have opposite signs.
        var antiSymmetric = new Matrix(new double[,] {
            {  0,  2, -1 },
            { -2,  0,  4 },
            {  1, -4,  0 }
        });

        // Break the skew-symmetry by putting a non-zero on the diagonal
        var brokenDiagonal = new Matrix(new double[,] {
            {  1,  2, -1 },
            { -2,  0,  4 },
            {  1, -4,  0 }
        });

        Assert.True(antiSymmetric.IsAntiSymmetric());
        Assert.False(brokenDiagonal.IsAntiSymmetric());
    }
    
    [Fact]
    public void GetDiagonal_SquareAndRectangular_ExtractsCorrectly()
    {
        // Arrange
        var square = new Matrix(new double[,] {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        var tall = new Matrix(new double[,] {
            { 10, 20 },
            { 30, 40 },
            { 50, 60 }
        });
        
        var wide = new Matrix(new double[,] {
            { 10, 20, 30 },
            { 40, 50, 60 }
        });

        // Act
        double[] sqDiag = square.GetDiagonal();
        double[] tallDiag = tall.GetDiagonal();
        double[] wideDiag = wide.GetDiagonal();

        // Assert
        Assert.Equal([1.0, 5.0, 9.0], sqDiag);
        Assert.Equal([10.0, 40.0], tallDiag);
        Assert.Equal([10.0, 50.0], wideDiag);
    }
    
    [Fact]
    public void Inverse_2x2Matrix_CalculatesCorrectly()
    {
        var m = new Matrix(new double[,] {
            { 4, 7 },
            { 2, 6 }
        });

        var inv = !m; // Using your shiny new ! operator
        var identity = m * inv; // A * A^-1 should equal I
        
        Assert.True(identity.IsCloseTo(Matrix.Identity(2)));
    }
    
    [Fact]
    public void Inverse_3x3Matrix_CalculatesCorrectly()
    {
        var m = new Matrix(new double[,] {
            { 1, 2, 3 },
            { 0, 1, 4 },
            { 5, 6, 0 }
        });

        var inv = m.Inverse();
        var identity = m * inv;

        Assert.True(identity.IsCloseTo(Matrix.Identity(3)));
    }
    
    [Fact]
    public void Inverse_4x4Matrix_CalculatesCorrectly()
    {
        var m = new Matrix(new double[,] {
            { 1, 2, 3, 4 },
            { 3, 3, 4, 11 },
            { 5, 6, 2, 3 },
            { 7, 8, 9, 12 }
        });

        var inv = m.Inverse();
        var identity = m * inv;

        Assert.True(identity.IsCloseTo(Matrix.Identity(4)));
    }
    
    [Fact]
    public void Inverse_SingularMatrix_ThrowsInvalidOperationException()
    {
        // Row 2 is just Row 1 multiplied by 2. This means the rows are linearly dependent, so det = 0.
        var m = new Matrix(new double[,] {
            { 1, 2 },
            { 2, 4 }
        });

        Assert.Throws<InvalidOperationException>(() => m.Inverse());
    }
    
    [Fact]
    public void Inverse_WithPreComputedDeterminant_CalculatesCorrectly()
    {
        var m = new Matrix(new double[,] {
            { 4, 7 },
            { 2, 6 }
        });

        // We pre-calculate it (or already know it is 10)
        double det = m.Determinant(); 

        // Act: Pass it into the fast-lane overload
        var inv = m.Inverse(knownDeterminant: det);
        var identity = m * inv;
        
        // Assert it still perfectly folds space
        Assert.True(identity.IsCloseTo(Matrix.Identity(2)));
    }
    
    [Fact]
    public void PseudoInverse_TallMatrix_CalculatesLeftInverse()
    {
        // Arrange: A 3x2 Tall Matrix
        var tall = new Matrix(new double[,] {
            { 1, 1 },
            { 1, 2 },
            { 1, 3 }
        });

        // Act
        var pseudoInv = tall.PseudoInverse();
        
        // Assert: A * A^+ * A must equal A
        var projection = tall * pseudoInv * tall;
        Assert.True(projection.IsCloseTo(tall));
    }
    
    [Fact]
    public void PseudoInverse_WideMatrix_CalculatesRightInverse()
    {
        // Arrange: A 2x3 Wide Matrix
        var wide = new Matrix(new double[,] {
            { 1, 1, 1 },
            { 1, 2, 3 }
        });

        // Act
        var pseudoInv = wide.PseudoInverse();
        
        // Assert: A * A^+ * A must equal A
        var projection = wide * pseudoInv * wide;
        Assert.True(projection.IsCloseTo(wide));
    }
    
    [Fact]
    public void PseudoInverse_SquareMatrix_FallsBackToStandardInverse()
    {
        // Arrange: A standard 2x2 Square Matrix
        var square = new Matrix(new double[,] {
            { 4, 7 },
            { 2, 6 }
        });

        // Act
        var pseudoInv = square.PseudoInverse();
        var standardInv = square.Inverse();

        // Assert: For invertible square matrices, the Pseudoinverse is identical to the Inverse
        Assert.True(pseudoInv.IsCloseTo(standardInv));
    }
    
    [Fact]
    public void Norm_L2_CalculatesEuclideanMagnitudeCorrectly()
    {
        // A standard 3-4-5 triangle represented as a vector
        var v = new Matrix(1, 2, [ 3.0, -4.0 ]);
        
        double l2 = v.Norm(NormType.L2);
        
        // Sqrt(3^2 + (-4)^2) = Sqrt(9 + 16) = Sqrt(25) = 5.0
        Assert.Equal(5.0, l2);
    }
    
    [Fact]
    public void Norm_L1_CalculatesManhattanMagnitudeCorrectly()
    {
        // A standard 3-4-5 triangle
        var v = new Matrix(1, 2, [ 3.0, -4.0 ]);
        
        double l1 = v.Norm(NormType.L1);
        
        // Abs(3) + Abs(-4) = 3 + 4 = 7.0
        Assert.Equal(7.0, l1);
    }
    
    [Fact]
    public void Norm_L2_Matrix_CalculatesFrobeniusNormCorrectly()
    {
        // For a 2D matrix, the L2 Norm is the Frobenius Norm
        var m = new Matrix(new double[,] {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // L2 is the default if no enum is provided
        double fNorm = m.Norm();

        // Sqrt(1 + 4 + 9 + 16) = Sqrt(30) ≈ 5.4772
        Assert.Equal(5.4772, fNorm, 4);
    }
    
    [Fact]
    public void Norm_L2_MassiveMatrix_UsesParallelReductionCorrectly()
    {
        // Arrange: 1000 x 200 = 200,000 elements. 
        // This exceeds the 100,000 threshold and triggers the Parallel path.
        var m = Matrix.Ones(1000, 200);

        // Act
        double l2 = m.Norm(NormType.L2);

        // Assert
        // The L2 norm of 200,000 ones is Sqrt(200,000)
        double expected = System.Math.Sqrt(200_000);
        Assert.Equal(expected, l2, 5); // Accurate to 5 decimal places
    }
    
    [Fact]
    public void Norm_L1_MassiveMatrix_UsesParallelReductionCorrectly()
    {
        // Arrange: 200,000 elements to trigger the Parallel path.
        // We multiply by -2.0 to ensure the absolute value logic works in parallel.
        var m = Matrix.Ones(1000, 200) * -2.0;

        // Act
        double l1 = m.Norm(NormType.L1);

        // Assert
        // The L1 norm is the sum of absolute values. 200,000 * |-2.0| = 400,000.
        Assert.Equal(400_000.0, l1);
    }
    
    [Fact]
    public void EuclideanDistance_CalculatesCorrectly()
    {
        var p1 = new Matrix(1, 2, [0.0, 0.0]);
        var p2 = new Matrix(1, 2, [3.0, 4.0]);

        // The distance from the origin to (3,4) is 5.0
        Assert.Equal(5.0, p1.EuclideanDistance(p2));
    }
    
    [Fact]
    public void CosineSimilarity_SameDirection_ReturnsOne()
    {
        var v1 = new Matrix(1, 2, [1.0, 1.0]);
        var v2 = new Matrix(1, 2, [5.0, 5.0]); // 5x longer, but same exact direction

        Assert.Equal(1.0, v1.CosineSimilarity(v2), 5);
    }
    
    [Fact]
    public void CosineSimilarity_SameDirection_ReturnsNegativeOne()
    {
        var v1 = new Matrix(1, 2, [-1.0, -1.0]);
        var v2 = new Matrix(1, 2, [5.0, 5.0]); // 5x longer, but same exact direction

        Assert.Equal(-1.0, v1.CosineSimilarity(v2), 5);
    }
    
    [Fact]
    public void CosineSimilarity_Orthogonal_ReturnsZero()
    {
        var v1 = new Matrix(1, 2, [1.0, 0.0]); // Pointing purely Right
        var v2 = new Matrix(1, 2, [0.0, 1.0]); // Pointing purely Up

        // 90-degree angle means 0 similarity
        Assert.Equal(0.0, v1.CosineSimilarity(v2), 5);
    }
    
    [Fact]
    public void DotProduct_SmallMatrices_CalculatesCorrectly()
    {
        // Arrange
        var v1 = new Matrix(1, 3, [1.0, 2.0, 3.0]);
        var v2 = new Matrix(1, 3, [4.0, -5.0, 6.0]);
        
        // Act
        double result = v1.DotProduct(v2);

        // Assert: (1 * 4) + (2 * -5) + (3 * 6) = 4 - 10 + 18 = 12.0
        Assert.Equal(12.0, result);
    }
    
    [Fact]
    public void DotProduct_MassiveMatrices_UsesParallelReductionCorrectly()
    {
        // Arrange: 1000 x 200 = 200,000 elements. 
        // This easily exceeds the 100,000 threshold, forcing the Parallel path.
        var m1 = Matrix.Ones(1000, 200) * 2.0; // Every element is 2.0
        var m2 = Matrix.Ones(1000, 200) * 3.0; // Every element is 3.0

        // Act
        double result = m1.DotProduct(m2);

        // Assert: Each element pair contributes 6.0 (2.0 * 3.0). 
        // 200,000 pairs * 6.0 = 1,200,000.0
        Assert.Equal(1_200_000.0, result);
    }
    
    [Fact]
    public void DotProduct_MismatchedSizes_ThrowsArgumentException()
    {
        // Arrange
        var v1 = new Matrix(1, 3); // 3 elements
        var v2 = new Matrix(1, 4); // 4 elements

        // Act & Assert
        Assert.Throws<ArgumentException>(() => v1.DotProduct(v2));
    }
    
    [Fact]
    public void OperatorAdd_MassiveMatrices_UsesParallelPathCorrectly()
    {
        // Arrange: 200,000 elements triggers the parallel path
        var m1 = Matrix.Ones(1000, 200) * 2.0; // Matrix of 2.0s
        var m2 = Matrix.Ones(1000, 200) * 3.0; // Matrix of 3.0s

        // Act
        var result = m1 + m2;

        // Assert: First, Middle, and Last elements should all be exactly 5.0
        Assert.Equal(5.0, result.Data[0]);
        Assert.Equal(5.0, result.Data[100_000]);
        Assert.Equal(5.0, result.Data[^1]);
    }
    
    [Fact]
    public void OperatorSubtract_MassiveMatrices_UsesParallelPathCorrectly()
    {
        // Arrange
        var m1 = Matrix.Ones(1000, 200) * 10.0; 
        var m2 = Matrix.Ones(1000, 200) * 4.0; 

        // Act
        var result = m1 - m2;

        // Assert
        Assert.Equal(6.0, result.Data[0]);
        Assert.Equal(6.0, result.Data[^1]);
    }
    
    [Fact]
    public void OperatorMultiplyScalar_MassiveMatrices_UsesParallelPathCorrectly()
    {
        // Arrange
        var m = Matrix.Ones(1000, 200) * 5.0; 

        // Act
        var result = m * 3.0;

        // Assert
        Assert.Equal(15.0, result.Data[0]);
        Assert.Equal(15.0, result.Data[^1]);
    }
    
    [Fact]
    public void HadamardProduct_CalculatesElementWiseMultiplicationCorrectly()
    {
        var m1 = new Matrix(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var m2 = new Matrix(new double[,] { { 2.0, 3.0 }, { 4.0, 5.0 } });

        var result = m1.HadamardProduct(m2);

        // Should be exactly 1*2, 2*3, 3*4, 4*5
        Assert.Equal(2.0, result[0, 0]);
        Assert.Equal(6.0, result[0, 1]);
        Assert.Equal(12.0, result[1, 0]);
        Assert.Equal(20.0, result[1, 1]);
    }
    
    [Fact]
    public void HadamardProduct_MassiveMatrices_UsesParallelPathCorrectly()
    {
        // Arrange: 200,000 elements triggers the parallel path
        var m1 = Matrix.Ones(1000, 200) * 2.0; 
        var m2 = Matrix.Ones(1000, 200) * 3.0; 

        // Act
        var result = m1.HadamardProduct(m2);

        // Assert
        Assert.Equal(6.0, result.Data[0]);
        Assert.Equal(6.0, result.Data[100_000]);
        Assert.Equal(6.0, result.Data[^1]);
    }
    
    [Fact]
    public void BroadcastAdd_AddsVectorToEveryRowCorrectly()
    {
        // Arrange: A standard 2x3 matrix
        var m = new Matrix(new double[,] { 
            { 1.0, 2.0, 3.0 }, 
            { 4.0, 5.0, 6.0 } 
        });
        
        // Arrange: A 1x3 vector (like a Neural Network bias)
        var bias = new Matrix(1, 3, [10.0, 100.0, 1000.0]);

        // Act
        var result = m.BroadcastAdd(bias);

        // Assert: The bias should have been added perfectly to both rows independently
        Assert.Equal(11.0, result[0, 0]); // 1 + 10
        Assert.Equal(102.0, result[0, 1]); // 2 + 100
        Assert.Equal(1003.0, result[0, 2]); // 3 + 1000

        Assert.Equal(14.0, result[1, 0]); // 4 + 10
        Assert.Equal(105.0, result[1, 1]); // 5 + 100
        Assert.Equal(1006.0, result[1, 2]); // 6 + 1000
    }
    
    [Fact]
    public void BroadcastAdd_InvalidDimensions_ThrowsArgumentException()
    {
        var m = Matrix.Zeros(4, 5);
        var invalidBias = Matrix.Zeros(1, 3); // Cols don't match
        var invalidShape = Matrix.Zeros(2, 5); // Not a 1D vector

        Assert.Throws<ArgumentException>(() => m.BroadcastAdd(invalidBias));
        Assert.Throws<ArgumentException>(() => m.BroadcastAdd(invalidShape));
    }
    
    [Fact]
    public void LUDecomposition_ValidSquareMatrix_DecomposesProperly()
    {
        // Arrange: Our standard 3x3 test matrix
        var a = new Matrix(new double[,] {
            {  2.0, -1.0, -2.0 },
            { -4.0,  6.0,  3.0 },
            { -4.0, -2.0,  8.0 }
        });

        // Act
        var (l, u) = a.LUDecomposition();
        var reconstructed = l * u;

        // Assert
        Assert.True(l.IsLowerTriangular());
        Assert.Equal(1.0, l[0, 0]);
        Assert.Equal(-2.0, l[1, 0]);
        Assert.Equal(1.0, l[1, 1]);
        Assert.Equal(-2.0, l[2, 0]);
        Assert.Equal(-1.0, l[2, 1]);
        Assert.Equal(1.0, l[2, 2]);
        
        Assert.True(u.IsUpperTriangular());
        Assert.Equal(2.0, u[0, 0]);
        Assert.Equal(-1.0, u[0, 1]);
        Assert.Equal(-2.0, u[0, 2]);
        Assert.Equal(4.0, u[1, 1]);
        Assert.Equal(-1.0, u[1, 2]);
        Assert.Equal(3.0, u[2, 2]);

        // L * U perfectly reconstructs the original matrix A
        Assert.True(reconstructed.IsCloseTo(a));
    }
    
    [Fact]
    public void LUDecomposition_ZeroPivot_ThrowsInvalidOperationException()
    {
        // Arrange: A matrix with a 0 in the top-left corner.
        // This will instantly crash the standard Doolittle algorithm because it divides by u[0,0].
        var a = new Matrix(new double[,] {
            { 0.0, 1.0 },
            { 1.0, 0.0 }
        });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => a.LUDecomposition());
    }
    
    [Fact]
    public void Determinant_LinearlyDependentRows_ReturnsZero()
    {
        // Arrange: Row 2 is exactly Row 1 multiplied by 2. This matrix is singular.
        var a = new Matrix(new double[,] {
            { 1.0, 2.0, 3.0 },
            { 2.0, 4.0, 6.0 }, 
            { 7.0, 8.0, 9.0 }
        });

        // Act
        var det = a.Determinant();

        // Assert: Determinant should be exactly 0.0
        Assert.Equal(0.0, det, 5); // Using 5 decimal places of precision
    }
    
    [Fact]
    public void PLUDecomposition_MassiveMatrix_TriggersParallelBulldozerAndDecomposesProperly()
    {
        // Arrange: Create a 200x200 matrix to cross the 128 remainingRows threshold
        var size = 200;
        var data = new double[size, size];
        var rand = new Random(42); // Fixed seed for reproducibility

        for (var i = 0; i < size; i++)
        {
            for (var j = 0; j < size; j++)
            {
                data[i, j] = rand.NextDouble() * 10.0;
            }
            // Make the matrix "diagonally dominant" to ensure it is highly stable
            data[i, i] += 100.0; 
        }

        var a = new Matrix(data);

        // Act: This will heavily utilize the Parallel.For loop
        var (p, l, u, _) = a.PLUDecomposition();
        
        var reconstructed = l * u;

        // Apply the permutation tracker P to our original matrix A
        // Because PLU guarantees P * A = L * U
        var permutedA = new Matrix(size, size);
        for (var i = 0; i < size; i++)
        {
            for (var j = 0; j < size; j++)
            {
                permutedA[i, j] = a[p[i], j];
            }
        }

        // Assert: L * U must perfectly match the Permuted A
        Assert.True(reconstructed.IsCloseTo(permutedA, 1e-8));
    }
    
    [Fact]
    public void PLUDecomposition_StandardMatrix_DecomposesAndTracksPermutations()
    {
        // Arrange: A matrix that will definitely require row swaps
        // Row 0 has a small diagonal (1.0), but Row 2 has a large number in that column (7.0)
        var a = new Matrix(3, 3, [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 0.0
        ]);

        // Act
        var (p, l, u, swaps) = a.PLUDecomposition();
        var reconstructed = l * u;

        // Manually apply the P array to our original matrix A
        var permutedA = new Matrix(3, 3);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++) permutedA[i, j] = a[p[i], j];
        }

        // Assert
        Assert.True(l.IsLowerTriangular());
        Assert.True(u.IsUpperTriangular());
        Assert.True(swaps > 0); // We guarantee at least one swap happened
        Assert.True(reconstructed.IsCloseTo(permutedA)); // P * A == L * U
    }
    
    [Fact]
    public void PLUDecomposition_SingularMatrix_DoesNotCrashAndYieldsZeroDeterminant()
    {
        // Arrange: Row 2 is exactly Row 1 multiplied by 2. 
        var a = new Matrix(3, 3, [
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
            7.0, 8.0, 9.0
        ]);

        // Act
        // It shouldn't throw an InvalidOperationException anymore
        var det = a.Determinant(); 

        // Assert
        Assert.Equal(0.0, det, 5);
    }
    
    [Fact]
    public void Solve_ValidSystemOfEquations_FindsCorrectVectorX()
    {
        // Arrange: A 3x3 system of equations
        // 2x +  y +  z = 8
        // -x +  y -  z = -3
        //  x + 2y + 3z = 8
        var a = new Matrix(3, 3, [
            2.0, 1.0, 1.0,
            -1.0, 1.0, -1.0,
            1.0, 2.0, 3.0
        ]);

        // CORRECTED TARGETS
        var b = new Matrix(3, 1, [8.0, -3.0, 8.0]);

        // Act
        var x = a.Solve(b);

        // Assert: Now the answers actually equal x=3, y=1, z=1
        Assert.Equal(3.0, x[0, 0], 5);
        Assert.Equal(1.0, x[1, 0], 5);
        Assert.Equal(1.0, x[2, 0], 5);
    }

    [Fact]
    public void Solve_MultipleRightHandSides_SolvesAllSimultaneously()
    {
        // Arrange: 
        var a = new Matrix(2, 2, [
            4.0, 3.0,
            6.0, 3.0
        ]);

        // We can pass a 2x2 target matrix to solve two different systems at once!
        var b = new Matrix(2, 2, [
            10.0, 1.0,
            12.0, 0.0
        ]);

        // Act
        var x = a.Solve(b);

        // Assert: Verify that A * X reconstructed perfectly equals B
        var reconstructedB = a * x;
        Assert.True(reconstructedB.IsCloseTo(b));
    }

    [Fact]
    public void Solve_DimensionMismatch_ThrowsArgumentException()
    {
        var a = Matrix.Identity(3);
        var b = new Matrix(4, 1); // 4 rows instead of 3

        Assert.Throws<ArgumentException>(() => a.Solve(b));
    }
    
    [Fact]
    public void QRDecomposition_SquareMatrix_DecomposesCorrectly()
    {
        // Arrange: A standard 3x3 matrix
        var a = new Matrix(3, 3, [
            12.0, -51.0,  4.0,
            6.0, 167.0, -68.0,
            -4.0,  24.0, -41.0
        ]);

        // Act
        var (q, r) = a.QRDecomposition();
        
        var reconstructed = q * r;
        var qtq = q.T * q; // Multiplying an orthogonal matrix by its transpose yields Identity

        // Assert
        Assert.True(r.IsUpperTriangular());
        Assert.True(reconstructed.IsCloseTo(a, 1e-8)); // Q * R == A
        Assert.True(qtq.IsCloseTo(Matrix.Identity(3), 1e-8)); // Q^T * Q == I
    }
    
    [Fact]
    public void QRDecomposition_TallMatrix_DecomposesCorrectly()
    {
        // Arrange: QR works on rectangular matrices too! (3 rows, 2 columns)
        var a = new Matrix(3, 2, [
            1.0, 2.0,
            0.0, 1.0,
            1.0, 0.0
        ]);

        // Act
        var (q, r) = a.QRDecomposition();
        
        var reconstructed = q * r;
        var qtq = q.T * q; 

        // Assert
        Assert.True(r.IsUpperTriangular());
        Assert.True(reconstructed.IsCloseTo(a, 1e-8));
        Assert.True(qtq.IsCloseTo(Matrix.Identity(3), 1e-8));
    }
    
    [Fact]
    public void QRDecomposition_LinearlyDependentColumns_HandlesGracefully()
    {
        // Arrange: Column 1 is exactly Column 0 multiplied by 2. 
        var a = new Matrix(3, 2, [
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0
        ]);

        // Act
        var (q, r) = a.QRDecomposition();
        var reconstructed = q * r;

        // Assert: The engine should decompose it perfectly without NaN corruption
        Assert.True(reconstructed.IsCloseTo(a, 1e-8));
    }
    
    [Fact]
    public void Eigenvalues_SymmetricMatrix_FindsCorrectValues()
    {
        // Arrange: A symmetric matrix
        var a = new Matrix(3, 3, [
            2.0, -1.0, 0.0,
            -1.0,  2.0, -1.0,
            0.0, -1.0,  2.0
        ]);

        // Act
        var eigenvalues = a.Eigenvalues();

        // Assert
        // The analytical eigenvalues for this specific matrix are:
        // 2 + sqrt(2), 2, and 2 - sqrt(2)
        // (approx 3.414, 2.0, 0.585)
        
        Assert.Contains(eigenvalues, v => Math.Abs(v - 3.41421356) < 1e-5);
        Assert.Contains(eigenvalues, v => Math.Abs(v - 2.0) < 1e-5);
        Assert.Contains(eigenvalues, v => Math.Abs(v - 0.58578643) < 1e-5);
    }
    
    [Fact]
    public void EigenDecomposition_SymmetricMatrix_ReconstructsPerfectly()
    {
        // Arrange
        var a = new Matrix(3, 3, [
            2.0, -1.0, 0.0,
            -1.0,  2.0, -1.0,
            0.0, -1.0,  2.0
        ]);

        // Act
        var (q, eigenvalues) = a.EigenDecomposition();
        
        // Build the Lambda matrix (diagonal matrix of eigenvalues)
        var lambda = Matrix.Zeros(3, 3);
        for (var i = 0; i < 3; i++) lambda[i, i] = eigenvalues[i];

        // Assert: A = Q * Lambda * Q^T
        var reconstructed = q * lambda * q.T;
        Assert.True(reconstructed.IsCloseTo(a, 1e-5));
    }
    
    [Fact]
    public void SVD_TallMatrix_ReconstructsPerfectly()
    {
        // Arrange: A standard 3x2 matrix
        var a = new Matrix(3, 2, [
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]);

        // Act
        var (u, s, v) = a.SVD();

        // Assert: A = U * S * V^T
        var reconstructed = u * s * v.T;
        Assert.True(reconstructed.IsCloseTo(a, 1e-5));
    }
    
    [Fact]
    public void PseudoInverse_LinearlyDependentMatrix_SucceedsViaSVD()
    {
        // Arrange: Row 2 is exactly Row 1 multiplied by 2. (Rank Deficient)
        // The old algebraic PseudoInverse threw an InvalidOperationException here.
        var brokenData = new Matrix(3, 2, [
            1.0, 2.0,
            2.0, 4.0,
            3.0, 6.0
        ]);

        // Act: The SVD engine will quietly filter out the bad dimension and solve it
        var pseudoInv = brokenData.PseudoInverse();

        // Assert: The geometric definition of a Moore-Penrose inverse is A * A^+ * A = A
        var projection = brokenData * pseudoInv * brokenData;
        Assert.True(projection.IsCloseTo(brokenData, 1e-5));
    }
}