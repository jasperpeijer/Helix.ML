namespace Helix.ML.Models;

public class TreeNode
{
    public bool IsLeaf { get; set; }
    public double PredictedClass { get; set; }
    public int FeatureIndex { get; set; }
    public double Threshold { get; set; }
    public TreeNode? Left { get; set; }
    public TreeNode? Right { get; set; }
}