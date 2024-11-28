public class Assignment2Part2 {
    public static void main(String[] args) {
        try {
            String[] sliceFiles = {
                "D:\\UVA\\CS6762\\final-project\\both-combined_slice1.csv",
                "D:\\UVA\\CS6762\\final-project\\both-combined_slice2.csv",
                "D:\\UVA\\CS6762\\final-project\\both-combined_slice3.csv",
                "D:\\UVA\\CS6762\\final-project\\both-combined_slice4.csv"
            };

            int[] featureIndices = new int[]{0, 1, 2, 3, 4, 5}; 

            for (String filePath : sliceFiles) {
                String[][] csvData = MyWekaUtils.readCSV(filePath);
                if (csvData == null) {
                    System.out.println("Failed to read CSV data from " + filePath);
                    continue;
                }

                String arffData = MyWekaUtils.csvToArff(csvData, featureIndices);

                // decision tree
                double accuracy = MyWekaUtils.classify(arffData, 1);
                System.out.println("Classification accuracy for " + filePath + ": " + accuracy + "%");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}