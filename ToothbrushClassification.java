import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;



import java.io.File;
import java.util.Random;

public class ToothbrushClassification { //this is the first version of everything
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("both-combined_slice4.csv")); 
        Instances data = loader.getDataSet();

        data.setClassIndex(data.numAttributes() - 1);

        int seed = 42;
        Random rand = new Random(seed);  
        Instances randData = new Instances(data);
        randData.randomize(rand);         // shuffle the data
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;

        Instances trainData = new Instances(randData, 0, trainSize);
        Instances testData = new Instances(randData, trainSize, testSize);

        Classifier randomForest = new RandomForest();
        Classifier svm = new SMO();
        MultilayerPerceptron neuralNetwork = new MultilayerPerceptron();//I need these parameter to build up my own NN
        System.out.println("Default MultilayerPerceptron Parameters:");
        System.out.println("Hidden Layers: " + neuralNetwork.getHiddenLayers());
        System.out.println("Learning Rate: " + neuralNetwork.getLearningRate());
        System.out.println("Momentum: " + neuralNetwork.getMomentum());
        System.out.println("Training Time: " + neuralNetwork.getTrainingTime());
        System.out.println("Validation Set Size: " + neuralNetwork.getValidationSetSize());
        System.out.println("Seed: " + neuralNetwork.getSeed());
        System.out.println("Debug Mode: " + neuralNetwork.getDebug());
        System.out.println("Normalization: " + neuralNetwork.getNormalizeAttributes());
        System.out.println("Decay Rate: " + neuralNetwork.getDecay());        

        Classifier knn = new IBk();
        AdaBoostM1 adaboost = new AdaBoostM1();
        adaboost.setClassifier(new DecisionStump());

        evaluateModel(randomForest, trainData, testData, "Random Forest");
        evaluateModel(svm, trainData, testData, "SVM");
        evaluateModel(neuralNetwork, trainData, testData, "Neural Network");
        evaluateModel(knn, trainData, testData, "KNN");
        evaluateModel(adaboost, trainData, testData, "Adaboost");

    }

    private static void evaluateModel(Classifier model, Instances trainData, Instances testData, String modelName) throws Exception {
        model.buildClassifier(trainData);

        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);

        System.out.println("Model: " + modelName);
        System.out.println("Accuracy: " + String.format("%.4f", eval.pctCorrect()) + "%");
        System.out.println("Confusion Matrix: ");
        double[][] confusionMatrix = eval.confusionMatrix();
        for (double[] row : confusionMatrix) {
            for (double val : row) {
                System.out.print(val + "\t");
            }
            System.out.println();
        }
        System.out.println("\nClassification Report: ");
        System.out.println(eval.toSummaryString());
        System.out.println("\n==================================\n");
    }
}
