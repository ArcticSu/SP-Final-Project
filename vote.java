import weka.classifiers.meta.Vote;
import weka.classifiers.trees.RandomForest;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class vote { //this method is not performing well
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

        //ensemble
        RandomForest randomForest = new RandomForest();
        SMO svm = new SMO();

        Vote vote = new Vote();
        vote.setClassifiers(new Classifier[]{randomForest, svm});
        evaluateModel(vote, trainData, testData, "Voting Ensemble");
    }

    private static void evaluateModel(Classifier model, Instances trainData, Instances testData, String modelName) throws Exception {
        model.buildClassifier(trainData);
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, testData);

        System.out.println("Model: " + modelName);
        System.out.println("Accuracy: " + eval.pctCorrect() + "%");
    }
}
