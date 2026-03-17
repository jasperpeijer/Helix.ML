namespace Helix.ML.Stat;

public static class Ranker
{
    /// <summary>
    /// Converts an array of raw values into their statistical ranks, 
    /// assigning fractional average ranks to tied values.
    /// </summary>
    public static double[] GetRanks(ReadOnlySpan<double> data)
    {
        int n = data.Length;
        
        if (n == 0) return [];

        double[] ranks = new double[n];
        var indexedData = new (double Value, int OriginalIndex)[n];

        for (int i = 0; i < n; i++)
        {
            indexedData[i] = (data[i], i);
        }
        
        Array.Sort(indexedData, (a, b) => a.Value.CompareTo(b.Value));

        for (int i = 0; i < n; i++)
        {
            int tieCount = 1;

            while (i + tieCount < n && indexedData[i].Value == indexedData[i + tieCount].Value)
            {
                tieCount++;
            }

            double startRank = i + 1;
            double endRank = i + tieCount;
            double averageRank = (startRank + endRank) / 2.0;

            for (int j = 0; j < tieCount; j++)
            {
                int originalPosition = indexedData[i + j].OriginalIndex;
                ranks[originalPosition] = averageRank;
            }
            
            i += tieCount - 1;
        }

        return ranks;
    }
}