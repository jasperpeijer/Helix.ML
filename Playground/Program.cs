using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Stat;
using ScottPlot;

public class Program
{
    public static void Main(string[] args)
    {
        // PCAExample();
        // DataFrameExample();
        LoadCsvExample();
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
        var models = new Column<string>("Model", ["Mustang", "Corolla", "Civic"]);
        var mpg = new Column<double>("MPG", [18.5, 32.1, 30.0]);
        var isElectric = new Column<bool>("IsElectric", [false, false, false]);

        var df = new DataFrame([models, mpg, isElectric]);

        df.Print();
        Console.WriteLine(df.Describe());
    }

    private static void LoadCsvExample()
    {
        var df = DataFrame.LoadCsv("datasets/df_test.csv");

        // Print the raw data to see the NaN and type handling
        Console.WriteLine(df);

        // Prove that Describe() mathematically ignores the 'Model' and 'IsElectric' columns,
        // successfully calculates the stats for 'Price', and skips over the Tesla's NaN 'EngineSize'.
        Console.WriteLine(df.Describe());
    }
}