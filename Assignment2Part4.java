
import java.util.Arrays;


public class Assignment2Part4 {

    private static double bestAccuracySoFar = 0;
    private static int[] allFeatureIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; 
    private static String[][] csvData;
    private static int[] bestFeatures = new int[0]; // initialize

    public static void main(String[] args) {
        try {
            String filePath = "C:\\Users\\student\\Documents\\SPMLC\\SP-Final-Project\\both-combined_slice4.csv"; 
            csvData = MyWekaUtils.readCSV(filePath);
            if (csvData == null) {
                System.out.println("Failed to read CSV data.");
                return;
            }

            double bestAccuracyImprovement = Double.MAX_VALUE;

            while (bestAccuracyImprovement > 0.01) {
                bestAccuracyImprovement = iterateFeatures();
            }

            System.out.println("Selected features: " + Arrays.toString(bestFeatures));
            System.out.println("Best accuracy: " + bestAccuracySoFar + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static double iterateFeatures() throws Exception {
        double bestAccuracyImprovement = -1;
        int bestFeatureIdx = -1;

        for (int i = 0; i < allFeatureIndices.length; i++) {
            if (contains(bestFeatures, allFeatureIndices[i])) continue;

            int[] newFeatureSet = combineFeatures(bestFeatures, allFeatureIndices[i]);
            double accuracy = classifyFeatures(newFeatureSet);
            double accuracyImprovement = accuracy - bestAccuracySoFar;

            if (accuracyImprovement > bestAccuracyImprovement) {
                bestAccuracyImprovement = accuracyImprovement;
                bestFeatureIdx = i;
            }
        }

        if (bestFeatureIdx != -1) {
            int[] newFeatureSet = combineFeatures(bestFeatures, allFeatureIndices[bestFeatureIdx]);
            double accuracy = classifyFeatures(newFeatureSet);
            if (accuracy > bestAccuracySoFar) {
                bestAccuracySoFar = accuracy;
                bestFeatures = newFeatureSet;
                System.out.println("New best accuracy: " + bestAccuracySoFar + "% with features " + Arrays.toString(bestFeatures));
            }
        }

        return bestAccuracyImprovement;
    }

    private static boolean contains(int[] array, int value) {
        for (int v : array) {
            if (v == value) return true;
        }
        return false;
    }

    private static int[] combineFeatures(int[] currentFeatures, int newFeature) {
        int[] newFeatureSet = new int[currentFeatures.length + 1];
        System.arraycopy(currentFeatures, 0, newFeatureSet, 0, currentFeatures.length);
        newFeatureSet[currentFeatures.length] = newFeature;
        return newFeatureSet;
    }

    private static double classifyFeatures(int[] featureIndices) throws Exception {
        String arffData = MyWekaUtils.csvToArff(csvData, featureIndices);
        return MyWekaUtils.classify(arffData, 5); // use decision tree classifier
    }
}