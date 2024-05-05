package com.example.mnist;

import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import au.com.bytecode.opencsv.CSVWriter;

//import android.support.v7.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {

    //IID Distribution
    private static final String mnistTrainUrl = "https://github.com/AyushSharma5583/Federated-Learning/blob/main/dataset/mnist_client2_iid.tar.gz?raw=true";

    private static final String mnistTestUrl = "https://github.com/AyushSharma5583/Federated-Learning/blob/main/dataset/mnist_test.tar.gz?raw=true";


    private static final String clientID = "2";

    double trainTime;
    TextView textTime;
    TextView textAccuracy;
    String trainDataset = "client2_mnist_iid_batch";

    //String testDataset = "client2_mnist_iid_batch_test";




    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);
        textTime = (TextView) findViewById(R.id.textView2);
        textAccuracy = (TextView) findViewById(R.id.textAccuracy);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
    }


    private class AsyncTaskRunner extends AsyncTask<String, Integer, String> {

        private MultiLayerNetwork trainedModel;

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("Started training...");
        }

        // This is our main background thread for training and testing the model.
        @Override
        protected String doInBackground(String... params) {

            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            double accuracy = 0;
            double precision = 0;
            double recall = 0;
            double f1_score = 0;


            try {

                // download the training dataset from the internet
                if (!new File(basePath + "/mnist_client2_iid").exists()) {
                    Log.d("Training Data downloaded", "Data downloaded from " + mnistTrainUrl);
                    File modelDir = new File(basePath + "/mnist_client2_iid");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTrainUrl, basePath)) {
                        DataUtilities.extractTarGz(basePath+"/mnist_client2_iid.tar.gz", basePath + "/mnist_client2_iid");
                    }
                }

                // download the testing dataset from the internet
                if (!new File(basePath + "/mnist_client2_iid_test").exists()) {
                    Log.d("Testing Data downloaded", "Data downloaded from " + mnistTestUrl);
                    File modelDir = new File(basePath + "/mnist_client2_iid_test");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTestUrl, basePath)) {
                        DataUtilities.extractTarGz(basePath+"/mnist_client2_iid.tar.gz", basePath + "/mnist_client2_iid_test");
                    }
                }




                // write the evaluation output to csv
                File file = new File(basePath + "/" + trainDataset + ".csv");
                FileWriter output = null;
                try {
                    output = new FileWriter(file);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                CSVWriter write = new CSVWriter(output);
                // Header column value
                String[] header = {"Accuracy", "Precision", "Recall", "F1 Score"};
                write.writeNext(header);


                // load the training dataset (Preprocessing)
                modelTrain model = new modelTrain();
                DataSetIterator mnistTrain = model.loadTrainData();


                // Generate the initial model
                modelBuildAndTrainAndEval initialModel = new modelBuildAndTrainAndEval();
                initialModel.modelGenerated();

                // Get the generated initial model
                String initialModelPath = basePath + "/initialModel.zip";
                MultiLayerNetwork modelLoad = ModelSerializer.restoreMultiLayerNetwork(initialModelPath);


                //Train the model
                model.modelTrain(modelLoad, mnistTrain);


                // Get the generated trained model
                String trainedModelPath = basePath+"/trainedModel" + clientID +".zip";
                trainedModel = ModelSerializer.restoreMultiLayerNetwork(trainedModelPath);
                Evaluation eval = initialModel.modelEval(trainedModel);


                accuracy = eval.accuracy();
                precision = eval.precision();
                recall = eval.recall();
                f1_score = eval.f1();
                String[] data = {String.valueOf(accuracy), String.valueOf(precision), String.valueOf(recall), String.valueOf(f1_score)};
                write.writeNext(data);
                write.close();



            }catch(Exception e){
                e.printStackTrace();
            }
            return "";
        }


        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }



        //This block executes in UI when background thread finishes

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("The time for training is "+ trainTime + "s");
        }



    }


    private  class modelTrain {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);
        int numEpochs = 1; // number of epochs to perform
        int numBatch = 468; // the number of batch to train for a epoch
        Random randBatch = new Random();

        private DataSetIterator loadTrainData() throws IOException, InterruptedException {

            // vectorization of train data
            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            File trainData = new File(basePath + "/mnist_client2_iid");

            FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
            ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
            trainRR.initialize(trainSplit);
            final DataSetIterator mnistTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

            // transform pixel values from 0-255 to 0-1 (min-max scaling)
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(mnistTrain);
            mnistTrain.setPreProcessor(scaler);

            return mnistTrain;
        }




        private void modelTrain(MultiLayerNetwork myNetwork,DataSetIterator mnistTrain) throws IOException {


            double startTime = System.nanoTime();
            Log.d("Training the model.", "Training the model....");


            // Training in epochs
            //            for(int l=0; l<numEpochs; l++) {
//                myNetwork.fit(mnistTrain);
//            }


           //  Training in batches
            for (int l = 0; l < numBatch; l++) {
                DataSet ds = mnistTrain.next();
                myNetwork.fit(ds);
            }


            //Training in one pass.
           // myNetwork.fit(mnistTrain, numEpochs);



            Log.d("Model trained", "Model has been trained.");
            trainTime = (System.nanoTime()-startTime)/1000000000;

            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            ModelSerializer.writeModel(myNetwork,  new File(basePath+"/trainedModel" + clientID +".zip"), true);


        }

    }



    private class modelBuildAndTrainAndEval {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);

        private void modelGenerated() throws IOException {

            Map<Integer, Double> learningRateSchedule = new HashMap<>();
            learningRateSchedule.put(0, 0.06);
            learningRateSchedule.put(200, 0.05);
            learningRateSchedule.put(400, 0.028);
            learningRateSchedule.put(600, 0.0060);
            learningRateSchedule.put(800, 0.001);

            ConvolutionLayer conv1 = new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                    .gradientNormalizationThreshold(1)
                    .build();

            SubsamplingLayer maxpool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            ConvolutionLayer conv2 = new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                    .gradientNormalizationThreshold(1)
                    .build();

            SubsamplingLayer maxpool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            DenseLayer fc = new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build();

            OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build();

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(rngSeed)
                    .l2(0.0005) // ridge regression value
                    .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image
                    .layer(conv1)
                    .layer(maxpool1)
                    .layer(conv2)
                    .layer(maxpool2)
                    .layer(fc)
                    .layer(outputLayer)
                    .build();


            MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
            myNetwork.init();
            myNetwork.setListeners(new ScoreIterationListener(10));

            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            ModelSerializer.writeModel(myNetwork, new File(basePath + "/initialModel.zip"), false);
            System.out.println("The initial model has been generated!");


        }


        private Evaluation modelEval(MultiLayerNetwork myNetwork) throws IOException, InterruptedException {

            // vectorization of test data
            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            File testData = new File(basePath + "/mnist_client2_iid_test");


            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
            testRR.initialize(testSplit);
            DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            mnistTest.setPreProcessor(scaler);


            Log.d("evaluate model", "Evaluating the model....");


            Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
            while (mnistTest.hasNext()) {
                DataSet next = mnistTest.next();
                INDArray output = myNetwork.output(next.getFeatures()); //get the networks prediction
                eval.eval(next.getLabels(), output); //check the prediction against the true class
            }

            //Since we used global variables to store the classification results, no need to return
            //a results string. If the results were returned here they would be passed to onPostExecute.
            Log.d("evaluate stats", eval.stats());
            Log.d("finished", "****************Example finished********************");
            return eval;
        }

    }

}