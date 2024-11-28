
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;


public class Assignment2Part5 {

    private static String[][] csvData;
    private static int[] allFeatureIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; 
    private static String filePath = "D:\\UVA\\CS6762\\final-project\\both-combined_slice4.csv"; 

    public static void main(String[] args) {
        try {
            csvData = MyWekaUtils.readCSV(filePath);
            if (csvData == null) {
                System.out.println("Failed to read CSV data.");
                return;
            }

            // decision tree
            performFeatureSelection(1);
            // random forest
            performFeatureSelection(2);
            // SVM
            performFeatureSelection(3);
            performFeatureSelection(4);
            performFeatureSelection(5);


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void performFeatureSelection(int classifierOption) throws Exception {
        double bestAccuracy = 0;
        Set<Integer> selectedFeatures = new HashSet<>();
        System.out.println("Starting feature selection with classifier option " + classifierOption);

        while (true) {
            double bestAccuracyImprovement = -1;
            int bestFeatureIdx = -1;

            for (int i = 0; i < allFeatureIndices.length; i++) {
                if (selectedFeatures.contains(allFeatureIndices[i])) continue;

                Set<Integer> newSelectedFeatures = new HashSet<>(selectedFeatures);
                newSelectedFeatures.add(allFeatureIndices[i]);

                double accuracy = evaluateFeatures(newSelectedFeatures, classifierOption);
                double accuracyImprovement = accuracy - bestAccuracy;

                if (accuracyImprovement > bestAccuracyImprovement) {
                    bestAccuracyImprovement = accuracyImprovement;
                    bestFeatureIdx = allFeatureIndices[i];
                }
            }

            if (bestFeatureIdx == -1) break; // No more improvements found

            selectedFeatures.add(bestFeatureIdx);
            bestAccuracy += bestAccuracyImprovement;

            System.out.println("Added feature " + bestFeatureIdx + " with accuracy improvement to " + bestAccuracy);
        }

        System.out.println("Final feature set for classifier option " + classifierOption + ": " +
                Arrays.toString(selectedFeatures.toArray(new Integer[0])) + " with accuracy " + bestAccuracy);
    }

    private static double evaluateFeatures(Set<Integer> featureIndices, int classifierOption) throws Exception {
        int[] intFeatureIndices = featureIndices.stream().mapToInt(i -> i).toArray();
        String arffData = MyWekaUtils.csvToArff(csvData, intFeatureIndices);
        return MyWekaUtils.classify(arffData, classifierOption);
    }
}