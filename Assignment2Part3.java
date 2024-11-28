public class Assignment2Part3 {
    public static void main(String[] args) {
        try {
            String filePath = "D:\\UVA\\CS6762\\final-project\\combined_slice4.csv"; 
            String[][] csvData = MyWekaUtils.readCSV(filePath);
            if (csvData == null) {
                System.out.println("Failed to read CSV data.");
                return;
            }

            int[] featureIndices = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; 

            String arffData = MyWekaUtils.csvToArff(csvData, featureIndices);

            // decision tree
            double accuracy = MyWekaUtils.classify(arffData, 1);
            System.out.println("Classification accuracy with decision tree and 12 features: " + accuracy + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}