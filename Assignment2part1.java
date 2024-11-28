import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class Assignment2part1 {
    public static void main(String[] args) {
        try {
            String[][] csvData = MyWekaUtils.readCSV("D:\\UVA\\CS6762\\final-project\\combined_slice1.csv");
            if (csvData == null) {
                System.out.println("Failed to read CSV data.");
                return;
            }

            int[] featureIndices = new int[]{0, 1, 2, 3, 4, 5}; 

            String arffData = MyWekaUtils.csvToArff(csvData, featureIndices);

            double accuracy = MyWekaUtils.classify(arffData, 1);
            System.out.println("Classification accuracy with decision tree: " + accuracy + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}