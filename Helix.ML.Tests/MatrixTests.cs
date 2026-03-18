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
            { 0, 1, 4, 11 },
            { 5, 6, 0, 3 },
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
}