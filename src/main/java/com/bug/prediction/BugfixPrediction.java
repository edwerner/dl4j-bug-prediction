package com.bug.prediction;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * This example is a version of the basic CSV example, but adds the following:
 * (a) Meta data tracking - i.e., where data for each example comes from
 * (b) Additional evaluation information - getting metadata for prediction errors
 *
 */
public class BugfixPrediction {

    public static void main(String[] args) throws  Exception {
        // Get dataset using record reader
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new ClassPathResource("bugfix-data.csv").getFile()));
        int batchSize = 10885;

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize);
        // Collect metadata and store it in DataSet objects
        iterator.setCollectMetaData(true);
        DataSet allData = iterator.next();
        allData.shuffle(123);
        
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(100);  

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
        List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);


        // Normalize data
        DataNormalization normalizer = new NormalizerStandardize();
        // Collect statistics (mean/stdev) from training data
        normalizer.fit(trainingData);        
        // Apply normalization to training data
        normalizer.transform(trainingData);
        // Apply normalization to test data from statistics from the training data
        normalizer.transform(testData);


        // Configure a simple model
        final int numInputs = 21;
        int outputNum = 21;

        long leftLimit = 10L;
        long rightLimit = 0L;
        long seed = leftLimit + (long) (Math.random() * (rightLimit - leftLimit));

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(new Sgd(0.1))
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //Fit the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<50; i++) {
            model.fit(trainingData);
        }

        // Evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        // Test set metadata
        eval.eval(testData.getLabels(), output, testMetaData);
        System.out.println(eval.stats());

        // Get a list of prediction errors from evaluation object
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        for(Prediction p : predictionErrors){
            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
                + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
        }

        // Load subset of the data to a DataSet object
        List<RecordMetaData> predictionErrorMetaData = new ArrayList<RecordMetaData>();
        for( Prediction p : predictionErrors ) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
        // Apply normalization to subset
        normalizer.transform(predictionErrorExamples);  

        // Load the raw data:
        List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);

        // Print out the prediction errors, along with raw data,
        // normalized data, labels and network predictions
        for(int i=0; i<predictionErrors.size(); i++ ){
            Prediction p = predictionErrors.get(i);
            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
            INDArray features = predictionErrorExamples.getFeatures().getRow(i);
            INDArray labels = predictionErrorExamples.getLabels().getRow(i);
            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

            INDArray networkPrediction = model.output(features);

            System.out.println(meta.getLocation() + ": "
                + "\tRaw Data: " + rawData
                + "\tNormalized: " + features
                + "\tLabels: " + labels
                + "\tPredictions: " + networkPrediction);
        }


        // Other evaluation methods:
        List<Prediction> list1 = eval.getPredictions(1,2);
        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);
        List<Prediction> list3 = eval.getPredictionsByActualClass(2);
        
        	System.out.println(list1.size());
        	System.out.println("***************************");
        	System.out.println(list2.size());
        	System.out.println("***************************");
        	System.out.println(list3.size());
        	System.out.println("***************************");

        	System.out.println("*********Get Predictions**********");
        	
        	for(Prediction p1 : list1) {
        		System.out.println(p1);
        	}

        	System.out.println("***Get Prediction by Predicted Class***");
        	
        	for(Prediction p2 : list2) {
        		System.out.println(p2);
        	}

        	System.out.println("***Get Predictions by Actual Class***");
        	
        	for(Prediction p3 : list3) {
        		System.out.println(p3);
        	}
       
    }
}
