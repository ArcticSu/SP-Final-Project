import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.Random;

public class feature_selection {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("combined_slice4.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Step 1: Feature Selection
        AttributeSelection selector = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(5); // Select top 5 features
        selector.setEvaluator(eval);
        selector.setSearch(ranker);
        selector.SelectAttributes(data);
        Instances selectedData = selector.reduceDimensionality(data);

        int seed = 42;
        Random rand = new Random(seed);
        Instances randData = new Instances(selectedData);
        randData.randomize(rand);
        int trainSize = (int) Math.round(randData.numInstances() * 0.8);
        Instances trainData = new Instances(randData, 0, trainSize);
        Instances testData = new Instances(randData, trainSize, randData.numInstances() - trainSize);

        Classifier randomForest = new RandomForest();
        Classifier svm = new SMO();
        Classifier neuralNetwork = new MultilayerPerceptron();

        evaluateModel(randomForest, trainData, testData, "Random Forest");
        evaluateModel(svm, trainData, testData, "SVM");
        evaluateModel(neuralNetwork, trainData, testData, "Neural Network");
    }

    private static void evaluateModel(Classifier model, Instances trainData, Instances testData, String modelName) throws Exception {
        model.buildClassifier(trainData);
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);

        System.out.println("Model: " + modelName);
        System.out.println("Accuracy: " + eval.pctCorrect() + "%");
        System.out.println("Confusion Matrix: ");
        double[][] confusionMatrix = eval.confusionMatrix();
        for (double[] row : confusionMatrix) {
            for (double val : row) {
                System.out.print(val + "\t");
            }
            System.out.println();
        }
        System.out.println(eval.toSummaryString());
    }
}
