using Helix.ML.Data;

namespace Helix.ML.Stat;

public class ClassificationMetrics
{
    public int TP { get; }
    public int TN { get; }
    public int FP { get; }
    public int FN { get; }
    public int Total => TP + TN + FP + FN;
    public double Accuracy => Total == 0 ? 0.0 : (double)(TP + TN) / Total;
    public double Precision => (TP + FP) == 0 ? 0.0 : (double)TP / (TP + FP);
    public double Recall => (TP + FN) == 0 ? 0.0 : (double)TP / (TP + FN);
    public double F1Score => (Precision + Recall) == 0 ? 0.0 : 2 * (Precision * Recall) / (Precision + Recall);
    
    private ClassificationMetrics(int tp, int tn, int fp, int fn)
    {
        TP = tp;
        TN = tn;
        FP = fp;
        FN = fn;
    }
    
    public static ClassificationMetrics Evaluate(double[] actual, double[] predicted, double positiveValue = 1.0)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Length mismatch between actual and predicted columns.");

        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (var i = 0; i < actual.Length; i++)
        {
            var isActualPos = actual[i] == positiveValue;
            var isPredPos = predicted[i] == positiveValue;

            if (isActualPos && isPredPos) tp++;
            else if (!isActualPos && !isPredPos) tn++;
            else if (!isActualPos && isPredPos) fp++;
            else if (isActualPos && !isPredPos) fn++;
        }

        return new ClassificationMetrics(tp, tn, fp, fn);
    }
    
    public void PrintReport(string positiveLabel)
    {
        Console.WriteLine("\n=== Classification Report ===");
        Console.WriteLine($"Accuracy:  {Accuracy:P2}");
        Console.WriteLine($"Precision: {Precision:P2} (When predicting '{positiveLabel}', it was right {Precision:P0} of the time)");
        Console.WriteLine($"Recall:    {Recall:P2} (It successfully caught {Recall:P0} of all actual '{positiveLabel}' cases)");
        Console.WriteLine($"F1-Score:  {F1Score:P2}");
        Console.WriteLine("\n--- Confusion Matrix ---");

        // Dogfooding: Using our own DataFrame to handle the perfect alignment formatting!
        string[] indices = [$"Actual '{positiveLabel}'", "Actual Other"];
        
        // We inject the raw TP/FP integers into the predicted positive column
        var predPosCol = new Column<int>($"Predicted '{positiveLabel}'", new int[] { TP, FP });
        
        // We inject the raw FN/TN integers into the predicted negative column
        var predOtherCol = new Column<int>("Predicted Other", new int[] { FN, TN });

        var matrixDf = new DataFrame(new IColumn[] { predPosCol, predOtherCol }, indices);
        
        matrixDf.Print(colWidth: 17, indexWidth: Math.Max($"Actual '{positiveLabel}'".Length, "Actual Other".Length) + 3);
        Console.WriteLine("=============================\n");
    }
}