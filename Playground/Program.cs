using Helix.ML.Data;
using Helix.ML.LinAlg;
using Helix.ML.Models;
using Helix.ML.Stat;
using ScottPlot;

public class Program
{
    public static async Task Main(string[] args)
    {
        // PCAExample();
        // DataFrameExample();
        // LoadCsvExample();
        // LinearRegressionExample();
        // OneHotEncodingExample();
        // await LinearRegressionWorkflowTest();
        // await FilteringExample();
        // await VectorMathAndFilteringTest();
        // await NewColumnExample();
        // KMeansExample();
        LogisticRegressionExample();
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
        
        Console.WriteLine(df.Info());
    }

    private static void LinearRegressionExample()
    {
        // 1. Load Data
        var df = DataFrame.LoadCsv("datasets/housing_data.csv");

        // 2. Split Data (80% Train, 20% Test)
        var (trainDf, testDf) = df.TrainTestSplit(testRatio: 0.2, seed: 42);

        // 3. Extract Matrices
        var xTrain = trainDf.ToMatrix("SquareFootage", "Bedrooms", "Age");
        var yTrain = trainDf.ToMatrix("Price");

        var xTest = testDf.ToMatrix("SquareFootage", "Bedrooms", "Age");
        var yTest = testDf.ToMatrix("Price");

        // 4. Train exclusively on the Training set
        var model = new LinearRegression();
        model.Fit(xTrain, yTrain);

        // 5. Grade the model on data it has never seen before!
        var (rmse, mse, r2) = model.Evaluate(xTest, yTest);
        Console.WriteLine($"MSE: {mse}");
        Console.WriteLine($"RMSE: {rmse}");
        Console.WriteLine($"R^2: {r2}");
    }

    private static void OneHotEncodingExample()
    {
        // 1. Load Data
        var df = DataFrame.LoadCsv("datasets/df_test.csv");

        // 2. Pre-process (Transform strings into math!)
        df = df.Encode("value_string");

        Console.WriteLine(df);
    }

    private static async Task LinearRegressionWorkflowTest()
    {
        // 1. DOWNLOAD REAL DATASET
        string url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv";
        string filePath = "datasets/taxis.csv";

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Downloading NYC Taxi Dataset...");
            using var client = new HttpClient();
            var csvData = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, csvData);
        }
        
        // 2. THE HELIX.ML PIPELINE
        Console.WriteLine("\nLoading and Inferring Data Types...");
        var df = DataFrame.LoadCsv(filePath);
        Console.WriteLine(df);
        Console.WriteLine(df.Info());
        Console.WriteLine(df.Describe());
        
        // Convert strings to One-Hot columns, and extract hours/days from datetimes
        Console.WriteLine("Encoding Categoricals and DateTimes...");
        df = df.Encode("color", "payment", "pickup");
        Console.WriteLine(df);
        Console.WriteLine(df.Info());
        Console.WriteLine(df.Describe());
        
        // 80/20 Train-Test Split
        Console.WriteLine("Splitting Dataset (80% Train, 20% Test)...");
        var (trainDf, testDf) = df.TrainTestSplit(testRatio: 0.2, seed: 42);
        
        // Extract the feature matrix and target matrix
        string[] features = ["distance", "fare", "tolls", "pickup_Hour", "color_yellow", "payment_credit card"];
        var xTrain = trainDf.ToMatrix(features);
        var yTrain = trainDf.ToMatrix("tip");
        
        var xTest = testDf.ToMatrix(features);
        var yTest = testDf.ToMatrix("tip");
        
        // 3. TRAIN AND EVALUATE
        Console.WriteLine("Training Linear Regression Model...");
        var model = new LinearRegression();
        model.Fit(xTrain, yTrain);
        
        var (rmse, mse, r2) = model.Evaluate(xTest, yTest);

        Console.WriteLine($"\n--- Helix.ML Results ---");
        Console.WriteLine($"RMSE: ${rmse:F4}");
        Console.WriteLine($"R-Squared: {r2:F4}");
        
        // 4. AUGMENT PREDICTIONS BACK TO THE DATAFRAME
        Console.WriteLine("\nAugmenting predictions onto the Test DataFrame...");
        var predictions = model.Predict(xTest);
        var predictionsDf = DataFrame.FromMatrix(predictions, ["Predicted_Tip"]);

        // Use the operator overload you built to horizontally fuse them!
        var finalDf = testDf | predictionsDf;


        // Print a slice of the final dataset showing the features, actual tip, and predicted tip
        Console.WriteLine(finalDf.Select("distance", "fare", "payment_credit card", "tip", "Predicted_Tip").ToString(15, 5));
    }

    private static async Task FilteringExample()
    {
        string url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv";
        string filePath = "datasets/taxis.csv";

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Downloading NYC Taxi Dataset...");
            using var client = new HttpClient();
            var csvData = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, csvData);
        }
        
        Console.WriteLine("\nLoading and Inferring Data Types...");
        var df = DataFrame.LoadCsv(filePath);
        Console.WriteLine(df);
        Console.WriteLine(df.Info());
        Console.WriteLine(df.Describe());

        var tip = df.GetColumn<double>("tip");
        var fare = df.GetColumn<double>("fare");
        var distance = df.GetColumn<double>("distance");
        var payment = df.GetColumn<string>("payment");
        
        var filteredDf = df.Filter(i => 
            (tip[i] > (fare[i] * 0.20) && payment[i] == "credit card") || distance[i] > 20.0
        );

        Console.WriteLine($"Found {filteredDf.Rows} rides matching this complex criteria!");
    }
    
    private static async Task VectorMathAndFilteringTest()
    {
        // ====================================================================
        // 1. DATA INGESTION
        // ====================================================================
        string url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv";
        string filePath = "datasets/taxis.csv";

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Downloading NYC Taxi Dataset...");
            using var client = new HttpClient();
            var csvData = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, csvData);
        }

        Console.WriteLine("\nLoading and Inferring Data Types...");
        var df = DataFrame.LoadCsv(filePath);
        Console.WriteLine(df.Info());
        
        // ====================================================================
        // 2. EXTRACT TYPED COLUMNS
        // ====================================================================
        var fare = df.GetColumn<double>("fare");
        var tip = df.GetColumn<double>("tip");
        var distance = df.GetColumn<double>("distance");
        var payment = df.GetColumn<string>("payment");
        var pickup = df.GetColumn<DateTime?>("pickup");
        var dropoff = df.GetColumn<DateTime?>("dropoff");

        // ====================================================================
        // 3. DATE MATH & DATAFRAME AUGMENTATION (|)
        // ====================================================================
        Console.WriteLine("\nCalculating Ride Durations...");
        
        // Subtract dropoff from pickup to get a Column<TimeSpan>
        var rideDurations = dropoff.SubtractDates(pickup);
        
        // Augment the new column onto the right side of the DataFrame
        df = df | new DataFrame([rideDurations]);
        Console.WriteLine($"Successfully added '{rideDurations.Name}' to DataFrame.");

        // ====================================================================
        // 4. VECTORIZED BOOLEAN MASKS (PANDAS STYLE)
        // ====================================================================
        Console.WriteLine("\n--- Executing Vectorized Filters ---");
        
        // Look at how clean this is! Native C# operators evaluating millions of cells in parallel.
        var isGenerous = tip > (fare * 0.20);
        var isCreditCard = payment == "credit card";
        
        // Chain the masks together using the overloaded Bitwise AND
        var generousCardMask = isGenerous & isCreditCard;
        
        var generousRidersDf = df.Filter(generousCardMask);
        Console.WriteLine($"Found {generousRidersDf.Rows} generous riders paying with a credit card.");
        
        // Print a slice to prove the math worked!
        generousRidersDf.Select("distance", "fare", "tip", "payment").Head(5);

        // ====================================================================
        // 5. CLOSURE-BASED FILTERING (C# PROPERTY ACCESS)
        // ====================================================================
        Console.WriteLine("\n--- Executing Closure-Based Filters ---");
        
        // We want to find rides that happened at night (after 8 PM) that went further than 10 miles.
        // Because we need the `.Hour` property of the DateTime, we use the Index Closure.
        var longNightRidesDf = df.Filter(i => 
            pickup[i].Value.Hour >= 20 && 
            distance[i] > 10.0
        );

        Console.WriteLine($"Found {longNightRidesDf.Rows} long night rides.");
        longNightRidesDf.Select("pickup", "distance", "fare", "tip").Head(5);
        
        // ====================================================================
        // 6. CHAINING IT ALL TOGETHER
        // ====================================================================
        Console.WriteLine("\n--- Extreme Filtering Pipeline ---");
        
        // Find people who paid cash, tipped $0, and rode for less than 2 miles.
        var cheapSkateDf = df
            .Filter(payment == "cash" & tip == 0.0 & distance < 2.0);

        Console.WriteLine($"Found {cheapSkateDf.Rows} short, cashless, zero-tip rides.");
        cheapSkateDf.Select("distance", "payment", "tip", "fare").Head(5);
    }

    private static async Task NewColumnExample()
    {
        string url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv";
        string filePath = "datasets/taxis.csv";

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Downloading NYC Taxi Dataset...");
            using var client = new HttpClient();
            var csvData = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, csvData);
        }

        Console.WriteLine("\nLoading and Inferring Data Types...");
        var df = DataFrame.LoadCsv(filePath);
        Console.WriteLine(df.Info());
        
        var fare = df.GetColumn<double>("fare");
        var tip = df.GetColumn<double>("tip");
        var pickup = df.GetColumn<DateTime?>("pickup");
        var dropoff = df.GetColumn<DateTime?>("dropoff");

        df["total_cost"] = fare + tip;
        df["ride_duration"] = dropoff.SubtractDates(pickup);
        df["pickup"] = pickup.Map(d => d?.TimeOfDay);
        df["dropoff"] = dropoff.Map(d => d?.TimeOfDay);

        var filteredDf = df.Select("pickup", "dropoff", "ride_duration", "fare", "tip", "total_cost");
        filteredDf.Print();
    }

    private static void KMeansExample()
    {
        var df = DataFrame.LoadCsv("datasets/knn_test.csv");
        df.Columns.Remove(df["id"]); 
        df.Head();

        var knn = new KNNClassifier(k: 3);
        knn.Fit(df, "premium_subscription", scaler: ScalerType.Standardize);
        Console.WriteLine("KNN Model successfully memorized and scaled 100 historical records.\n");

        var ageCol = new Column<double>("age", [23, 55, 30, 48]);
        var annualIncomeCol = new Column<double>("annual_income", [32000, 150000, 50000, 120000]);
        var newCustomersDf = new DataFrame([ageCol, annualIncomeCol]);
        newCustomersDf.Head();
        
        var predictions = knn.Predict(newCustomersDf);
        newCustomersDf["Predicted_Class"] = predictions;

        Console.WriteLine("Predictions for new customers:");
        Console.WriteLine(newCustomersDf.ToString());
    }

    private static void LogisticRegressionExample()
    {
        // 1. Load the 200-record dataset
        var df = DataFrame.LoadCsv("datasets/logreg_test.csv");

        // 2. Drop the ID column so it doesn't skew the gradient descent
        df.Columns.Remove(df["id"]);

        // 3. Initialize Logistic Regression
        // A learning rate of 0.1 and 2000 iterations gives the math plenty of time 
        // to slide down the error curve and lock into the perfect weights.
        var logReg = new LogisticRegression(learningRate: 0.1, maxIterations: 2000);

        Console.WriteLine("Training Logistic Regression model... (Performing 2000 iterations of Gradient Descent)");

        // 4. Train with Standardization
        // (Mandatory, otherwise Income will overpower Age during training)
        logReg.Fit(df, targetColumn: "premium_subscription", positiveLabel: "Yes", scalerType: ScalerType.Standardize);

        Console.WriteLine("Training complete. Weights and Bias are locked in.\n");

        // 5. Test on completely unseen data profiles

        // Profile 1: Young, Low Income (Should definitely be No)
        // Profile 2: Middle-aged, Solid Income (Borderline, leaning Yes)
        // Profile 3: Senior, High Income (Should definitely be Yes)
        // Profile 4: Young but High Income (A tricky outlier!)
        var ageCol = new Column<double>("age", new double[] { 22, 45, 60, 25 });
        var annualIncomeCol = new Column<double>("annual_income", new double[] { 35000, 110000, 180000, 140000 });
        var testDf = new DataFrame([ageCol, annualIncomeCol]);

        // 6. Predict probabilities and human-readable classes
        var predictions = logReg.Predict(testDf, threshold: 0.5);
        testDf["Predicted_Premium"] = predictions;

        // Let's also attach the raw probabilities so you can see exactly how confident the model is
        var rawProbabilities = logReg.PredictProbabilities(testDf);
        testDf["Confidence"] = new Column<double>("Confidence", rawProbabilities);

        Console.WriteLine("Predictions for unseen customers:");
        Console.WriteLine(testDf.ToString());
        
        // Access individual metrics...
        Console.WriteLine($"Accuracy: {logReg.TrainingMetrics!.Accuracy:P2}");
        Console.WriteLine($"F1 Score: {logReg.TrainingMetrics!.F1Score:P2}");

        // ...or print the whole report!
        logReg.TrainingMetrics.PrintReport("Yes");
    }
}