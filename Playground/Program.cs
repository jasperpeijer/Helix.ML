using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Stat;
using ScottPlot;

public class Program
{
    public static void Main(string[] args)
    {
        // PCAExample();
        DataFrameExample();
    }

    private static void PCAExample()
    {
        // 1. GENERATE THE "REALISTIC" DATASET
        int numSamples = 500;
        int numFeatures = 50;
        int trueDimensions = 5;

        var rand = new Random(42);
        var data = new Matrix(numSamples, numFeatures);

        // Step A: Create 5 "Hidden" true traits for 500 samples
        double[][] hiddenTraits = new double[numSamples][];
        for (int i = 0; i < numSamples; i++)
        {
            hiddenTraits[i] = new double[trueDimensions];
            for (int k = 0; k < trueDimensions; k++)
                hiddenTraits[i][k] = rand.NextDouble() * 10;
        }

        // Step B: Create a random 5x50 projection matrix to smear the 5 traits across 50 columns
        var projection = new Matrix(trueDimensions, numFeatures);
        for (int k = 0; k < trueDimensions; k++)
            for (int j = 0; j < numFeatures; j++)
                projection[k, j] = (rand.NextDouble() - 0.5) * 2; 

        // Step C: Build the 50-feature dataset (Hidden Traits * Projection) + Random Noise
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                double value = 0;
                for (int k = 0; k < trueDimensions; k++)
                {
                    value += hiddenTraits[i][k] * projection[k, j];
                }
                data[i, j] = value + (rand.NextDouble() - 0.5); // Add noise
            }
        }

        Console.WriteLine($"Generated dataset with {numSamples} rows and {numFeatures} columns.");

        // 2. RUN HELIX.ML PCA
        var pca = new PCA();
        pca.Fit(data, maxIterations: 30000, tolerance: 1e-7);

        // 3. CALCULATE CUMULATIVE VARIANCE
        double[] cumulativeVariance = new double[numFeatures];
        double sum = 0;
        for (int i = 0; i < numFeatures; i++)
        {
            sum += pca.ExplainedVarianceRatio[i];
            cumulativeVariance[i] = sum;
        }

        Console.WriteLine("\n--- Top 10 Principal Components ---");
        for(int i = 0; i < 10; i++)
        {
            Console.WriteLine($"PC{i+1}: {pca.ExplainedVarianceRatio[i]*100:F2}% (Cumulative: {cumulativeVariance[i]*100:F2}%)");
        }

        // 4. VISUALIZATION: THE SCREE PLOT
        var plt = new Plot();
        plt.Title("PCA Scree Plot: Discovering Hidden Dimensions");
        plt.XLabel("Principal Component Index");
        plt.YLabel("Cumulative Explained Variance (%)");

        // Plot the cumulative variance curve
        double[] xs = new double[numFeatures];
        for(int i = 0; i < numFeatures; i++) xs[i] = i + 1;

        var scatter = plt.Add.Scatter(xs, cumulativeVariance);
        scatter.LineWidth = 2;
        scatter.MarkerSize = 5;

        // Draw a red dotted line at the 95% threshold
        var targetLine = plt.Add.Line(1, 0.95, 50, 0.95);
        targetLine.Color = Colors.Red;
        targetLine.LineWidth = 2;
        targetLine.LinePattern = LinePattern.Dashed;

        plt.SavePng("scree_plot.png", 800, 500);
        Console.WriteLine("\nSuccess! Saved Scree Plot to scree_plot.png");
    }

    private static void DataFrameExample()
    {
        // Create raw data
        double[] rawCarData = [
            21.0, 160.0, 2.62,
            21.0, 160.0, 2.875,
            22.8, 108.0, 2.32,
            21.4, 258.0, 3.215,
            18.7, 360.0, 3.44
        ];
        var matrix = new Matrix(5, 3, rawCarData);

        // Wrap it in a DataFrame
        var df = new DataFrame(matrix, ["MPG", "Horsepower", "Weight"]);

        // View the data exactly like pandas df.head()
        df.Head();

        // Slice it! Just grab HP and Weight for PCA
        var featuresDf = df.Select("Horsepower", "Weight");
        featuresDf.Head(3);

        // Extract a single column for math (Returns an Nx1 Matrix)
        Matrix y = df["MPG"];
    }
}