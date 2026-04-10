using Helix.ML.Data;
using Helix.ML.LinAlg;

namespace Helix.ML.Models;

public class DecisionTree(int maxDepth = 5)
{
    public TreeNode? Root { get; private set; }
    public int MaxDepth { get; } = maxDepth;
    private Dictionary<string, double>? _labelToDouble;
    private Dictionary<double, string>? _doubleToLabel;

    private double CalculateGini(double[] y)
    {
        if (y.Length == 0) return 0.0;

        var classCounts = new Dictionary<double, int>();

        for (var i = 0; i < y.Length; i++)
        {
            if (classCounts.TryGetValue(y[i], out var count))
                classCounts[y[i]] = count + 1;
            else
                classCounts[y[i]] = 1;
        }

        var impurity = 1.0;
        double totalRows = y.Length;

        foreach (var count in classCounts.Values)
        {
            var probability = count / totalRows;
            impurity -= (probability * probability);
        }

        return impurity;
    }

    public void Fit(DataFrame df, string targetColumn)
    {
        var trainingDf = df.Clone();
        var yCol = trainingDf[targetColumn];
        trainingDf.Columns.Remove(yCol);

        _labelToDouble = new Dictionary<string, double>();
        _doubleToLabel = new Dictionary<double, string>();
        var currentLabelId = 0.0;
        var yDouble = new double[yCol.Length];

        for (var i = 0; i < yCol.Length; i++)
        {
            var label = yCol.GetValue(i)?.ToString() ?? "";

            if (!_labelToDouble.ContainsKey(label))
            {
                _labelToDouble[label] = currentLabelId;
                _doubleToLabel[currentLabelId] = label;
                currentLabelId++;
            }
            
            yDouble[i] = _labelToDouble[label];
        }
        
        Fit(trainingDf.ToMatrix(), yDouble);
    }

    public void Fit(Matrix x, double[] y)
    {
        Root = BuildTree(x, y, 0);
    }
    
    private TreeNode BuildTree(Matrix x, double[] y, int depth)
    {
        if (depth >= MaxDepth || y.Length == 0 || CalculateGini(y) == 0.0)
        {
            return new TreeNode { IsLeaf = true, PredictedClass = GetMajorityClass(y) };
        }

        var bestSplit = FindBestSplit(x, y);

        if (bestSplit.FeatureIndex == -1)
        {
            return new TreeNode { IsLeaf =  true, PredictedClass = GetMajorityClass(y) };
        }
        
        var (leftX, leftY, rightX, rightY) = SplitData(x, y, bestSplit.FeatureIndex, bestSplit.Threshold);

        return new TreeNode
        {
            IsLeaf = false,
            FeatureIndex = bestSplit.FeatureIndex,
            Threshold = bestSplit.Threshold,
            Left = BuildTree(leftX, leftY, depth + 1),
            Right = BuildTree(rightX, rightY, depth + 1),
        };
    }

    private (int FeatureIndex, double Threshold) FindBestSplit(Matrix x, double[] y)
    {
        var bestFeature = -1;
        var lowestGini = double.MaxValue;
        var (rows, cols) = (x.Rows, x.Cols);
        double bestThreshold = 0;

        for (var col = 0; col < cols; col++)
        {
            var uniqueValues = new HashSet<double>();

            for (var r = 0; r < rows; r++) uniqueValues.Add(x[r, col]);

            foreach (var threshold in uniqueValues)
            {
                var (leftY, rightY) = SplitLabelsOnly(x, y, col, threshold);

                if (leftY.Count == 0 || rightY.Count == 0) continue;

                var leftWeight = (double)leftY.Count / rows;
                var rightWeight = (double)rightY.Count / rows;
                var currentGini = (leftWeight * CalculateGini(leftY.ToArray())) + (rightWeight * CalculateGini(rightY.ToArray()));

                if (currentGini < lowestGini)
                {
                    lowestGini = currentGini;
                    bestFeature = col;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeature, bestThreshold);
    }

    private double GetMajorityClass(double[] y)
    {
        var counts = new Dictionary<double, int>();
        var majorityClass = y[0];
        var maxCount = 0;

        for (var i = 0; i < y.Length; i++)
        {
            if (!counts.ContainsKey(y[i])) counts[y[i]] = 0;
            counts[y[i]]++;

            if (counts[y[i]] > maxCount)
            {
                maxCount = counts[y[i]];
                majorityClass = y[i];
            }
        }

        return majorityClass;
    }

    private (List<double> leftY, List<double> rightY) SplitLabelsOnly(Matrix x, double[] y, int col, double threshold)
    {
        var leftY = new List<double>(y.Length);
        var rightY = new List<double>(y.Length);

        for (var i = 0; i < x.Rows; i++)
        {
            if (x[i, col] <= threshold) leftY.Add(y[i]);
            else rightY.Add(y[i]);
        }
        
        return (leftY, rightY);
    }

    private (Matrix leftX, double[] leftY, Matrix rightX, double[] rightY) SplitData(Matrix x, double[] y, int col, double threshold)
    {
        var (leftIndices, rightIndices) = (new List<int>(), new List<int>());

        for (var i = 0; i < x.Rows; i++)
        {
            if (x[i, col] <= threshold) leftIndices.Add(i);
            else rightIndices.Add(i);
        }

        var cols = x.Cols;
        var leftX = new Matrix(leftIndices.Count, cols);
        var rightX = new Matrix(rightIndices.Count, cols);
        var leftY = new double[leftIndices.Count];
        var rightY = new double[rightIndices.Count];

        for (var i = 0; i < leftIndices.Count; i++)
        {
            leftY[i] = y[leftIndices[i]];
            
            for (var j = 0; j < cols; j++) leftX[i, j] = x[leftIndices[i], j];
        }

        for (var i = 0; i < rightIndices.Count; i++)
        {
            rightY[i] = y[rightIndices[i]];
            
            for (var j = 0; j < cols; j++) rightX[j, i] = x[rightIndices[i], j];
        }
        
        return (leftX, leftY, rightX, rightY);
    }

    public Column<string> Predict(DataFrame df)
    {
        if (Root == null || _doubleToLabel == null)
            throw new InvalidOperationException("Tree must be fitted before predicting.");

        var matrix = df.ToMatrix();
        var numericPredictions = Predict(matrix);
        var stringPredictions = new string[numericPredictions.Length];

        Parallel.For(0, numericPredictions.Length, i =>
        {
            stringPredictions[i] = _doubleToLabel[numericPredictions[i]];
        });

        return new Column<string>("Predicted_Class", stringPredictions);
    }

    public double[] Predict(Matrix x)
    {
        if (Root == null)
            throw new InvalidOperationException("Tree must be fitted before predicting.");

        var predictions = new double[x.Rows];

        Parallel.For(0, x.Rows, i =>
        {
            predictions[i] = PredictRow(x, i, Root);
        });

        return predictions;
    }

    private double PredictRow(Matrix x, int row, TreeNode node)
    {
        if (node.IsLeaf) return node.PredictedClass;

        if (x[row, node.FeatureIndex] <= node.Threshold) return PredictRow(x, row, node.Left!);
        else return PredictRow(x, row, node.Right!);
    }
}