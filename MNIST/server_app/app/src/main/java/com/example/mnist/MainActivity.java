package com.example.mnist;

import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Debug;
import android.os.Environment;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.Console;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

import android.util.Log;

import java.net.Socket;
import java.util.Set;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import au.com.bytecode.opencsv.CSVWriter;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;


public class MainActivity extends AppCompatActivity {
    //private static final String basePath = Environment.getExternalStorageDirectory() + "/server";


    private static final String mnistTesturl = "https://github.com/AyushSharma5583/Federated-Learning/blob/main/dataset/mnist_test.tar.gz?raw=true";

    TextView text;
    double targetAccuracy = 0.90;
    int trainNum = 0;
    String trainDataset = "mnist_iid";



    ///Secret key for Caeser Cipher
    //private static final int ENCRYPTION_KEY = 3;


    ////Secret key for AES-128 ECB and CBC mode
//    private static final byte[] KEY_VALUE = {
//            (byte) 0x01, (byte) 0x23, (byte) 0x45, (byte) 0x67,
//            (byte) 0x89, (byte) 0xab, (byte) 0xcd, (byte) 0xef,
//            (byte) 0xfe, (byte) 0xdc, (byte) 0xba, (byte) 0x98,
//            (byte) 0x76, (byte) 0x54, (byte) 0x32, (byte) 0x10
//    };


    ////Secret key for AES-256 GCM mode
    private static final byte[] KEY_VALUE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};



    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);
        text = (TextView) findViewById(R.id.textView2);
        String basePath = getCacheDir().getAbsolutePath() + "/server";
        File serverDir = new File(basePath);
        if (!serverDir.exists()) {
            serverDir.mkdirs();
        }


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
        // Runs in UI before background thread is called

        private MultiLayerNetwork updatedModel;

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            text.setText("Generated the initial model and sent to clients...");
        }

        // This is our main background thread for training the model and uploading the model
        @Override
        protected String doInBackground(String... params) {
            String basePath = getCacheDir().getAbsolutePath() + "/server";
            final int numClient = 2;
            int numConnectedClient = 0;
            double accuracy = 0;
            double precision = 0;
            double recall = 0;
            double f1_score = 0;
            int num = 1;

            try {
                // Downloading the dataset from the Internet
                if (!new File(basePath + "/mnist_test").exists()) {
                    Log.d("Data download", "Data downloaded from " + mnistTesturl);
                    File modelDir = new File(basePath + "/mnist_test");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTesturl, basePath)) {
                        DataUtilities.extractTarGz(basePath + "/mnist_test.tar.gz", basePath + "/mnist_test");
                    }
                }

                // set up the server socket and set the port as 5000
                ServerSocket ss = new ServerSocket(5000);
                System.out.println("ServerSocket awaiting connections...");

                // define the thread for receiving the message from multiple clients
                Thread[] thread = new Thread[numClient];
                Thread[] initModelThread = new Thread[numClient];
                Thread[] signalThread = new Thread[numClient];

                modelBuildAndTrainAndEval model = new modelBuildAndTrainAndEval();
                // generate the initial model
                model.modelGenrated();

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
                String[] header = {"ID", "Accuracy", "Precision", "Recall", "F1 Score"};
                write.writeNext(header);

                // The training process
                // First the server sends the model to all the clients then waiting for the trained model from the clients
                // Once receiving the model from all the clients, do the aggregation and evaluate the aggregated model
                // Evaluate the model to see if it reaches the target accuracy
                while (true) {
                    while (numConnectedClient < numClient) {
                        Socket socketInit = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                        System.out.println("Connection from " + socketInit + "!");
                        numConnectedClient = numConnectedClient + 1;
                        initModelThread[numConnectedClient - 1] = new Thread(new ServiceSend(socketInit));
                        initModelThread[numConnectedClient - 1].start();
                    }
                    numConnectedClient = 0;
                    Thread.sleep(1000);
                    System.out.println("Waiting the trained model from clients...");
                    while (numConnectedClient < numClient) {
                        Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                        System.out.println("Connection from " + socket + "!");
                        numConnectedClient = numConnectedClient + 1;
                        thread[numConnectedClient - 1] = new Thread(new ServiceReceive(socket));
                        thread[numConnectedClient - 1].start();
                        Thread.sleep(5000);
                    }
                    String updatedModelPath_cID_1 = basePath + "/localModel_cID_1.zip";
                    String updatedModelPath_cID_2 = basePath + "/localModel_cID_2.zip";


                    while (new File(updatedModelPath_cID_1).length() < 300000 || new File(updatedModelPath_cID_2).length() < 300000) {
                        Thread.sleep(2000);
                    }
                    Thread.sleep(5000);


                    // Decrypt the encrypted models
                    decryptModelFile(new File(updatedModelPath_cID_1));
                    decryptModelFile(new File(updatedModelPath_cID_2));


                    double startTime = System.nanoTime();
                    MultiLayerNetwork modelLoad_cID_1 = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath_cID_1);
                    MultiLayerNetwork modelLoad_cID_2 = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath_cID_2);

                    weightAveraging(modelLoad_cID_1, modelLoad_cID_2);
                    String updatedModelPath = basePath + "/updatedModel.zip";
                    updatedModel = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath);
                    Evaluation eval = model.modelEval(updatedModel);
                    double aggragateAndEvalTime = (System.nanoTime() - startTime) / 1000000000;
                    System.out.println("The time for evalation is " + aggragateAndEvalTime + " s");

                    accuracy = eval.accuracy();
                    precision = eval.precision();
                    recall = eval.recall();
                    f1_score = eval.f1();
                    String[] data = {String.valueOf(num), String.valueOf(accuracy), String.valueOf(precision), String.valueOf(recall), String.valueOf(f1_score)};
                    write.writeNext(data);
                    num = num + 1;

                    numConnectedClient = 0;
                    thread[0].interrupt();
                    thread[1].interrupt();
                    trainNum = trainNum + 1;
                    if (accuracy >= targetAccuracy) {
                        System.out.println("Save the metrics output");
                        write.close();
                        while (numConnectedClient < numClient) {
                            Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                            System.out.println("Connection from " + socket + "!");
                            numConnectedClient = numConnectedClient + 1;
                            signalThread[numConnectedClient - 1] = new Thread(new stopSignalSend(socket));
                            signalThread[numConnectedClient - 1].start();
                        }
                        break;

                    } else {
                        while (numConnectedClient < numClient) {
                            Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                            System.out.println("Connection from " + socket + "!");
                            numConnectedClient = numConnectedClient + 1;
                            signalThread[numConnectedClient - 1] = new Thread(new ContinueSignalSend(socket));
                            signalThread[numConnectedClient - 1].start();
                        }
                        numConnectedClient = 0;
                    }

                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return "";
        }

        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {

            super.onProgressUpdate(values);
        }



        /*
        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            text.setText("The accuracy reach the target " + targetAccuracy + "\n" + "It has taken " + trainNum + " times of training iteration");

        }

        */




//        @Override
//        protected void onPostExecute(String result) {
//            super.onPostExecute(result);
//            // Hide the progress bar now that we are finished
//            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
//            bar.setVisibility(View.INVISIBLE);
//            text.setText("The accuracy reach the target " + targetAccuracy + "\n" + "It has taken " + trainNum + " times of training iteration");
//
//            Button inferenceButton = findViewById(R.id.inferenceButton);
//
//            // Make the inference button visible after training
//            inferenceButton.setVisibility(View.VISIBLE);
//
//            inferenceButton.setOnClickListener(new View.OnClickListener() {
//                @Override
//                public void onClick(View v) {
//                    bar.setVisibility(View.VISIBLE);
//                    // To prevent Main thread blocking
//                    new Thread(new Runnable() {
//                        @Override
//                        public void run() {
//                            try {
//                                ServerSocket serverSocket = new ServerSocket(5001);
//
//
//                                    // Server socket creation and connection acceptance in the new thread
//                                    //ServerSocket serverSocket = new ServerSocket(5001);
//                                    Socket socket = serverSocket.accept();
//                                    System.out.println("Connection from " + socket + "!");
//
//                                    // Call the receiveFile function here
//                                    File imageFile = receiveFile(socket);
//
//                                    System.out.print("Image Received");
//
//                                    // Preprocess the image and classify it
//                                    int predictedClass = classifyImage(updatedModel, imageFile);
//                                    System.out.println("Predicted class: " + predictedClass);
//
//
//                                    // Send the predicted class back to the client
//                                    PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
//                                    out.println(predictedClass);
//
//
//                                    bar.setVisibility(View.INVISIBLE);
//                                    //socket.close();
//                                    //serverSocket.close();
//
//
//                            } catch (Exception e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    }).start();
//                }
//            });
//        }


















//        protected void onPostExecute(String result) {
//            super.onPostExecute(result);
//            // Hide the progress bar now that we are finished
//            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
//            bar.setVisibility(View.INVISIBLE);
//            text.setText("The accuracy reach the target " + targetAccuracy + "\n" + "It has taken " + trainNum + " times of training iteration");
//
//            Button inferenceButton = findViewById(R.id.inferenceButton);
//
//            // Make the inference button visible after training
//            inferenceButton.setVisibility(View.VISIBLE);
//
//            inferenceButton.setOnClickListener(new View.OnClickListener() {
//                @Override
//                public void onClick(View v) {
//                    bar.setVisibility(View.VISIBLE);
//                    // To prevent Main thread blocking
//                    new Thread(new Runnable() {
//                        @Override
//                        public void run() {
//                            try {
//                                ServerSocket serverSocket = new ServerSocket(5001);
//
//
//                                // Server socket creation and connection acceptance in the new thread
//
//                                Socket socket = serverSocket.accept();
//                                System.out.println("Connection from " + socket + "!");
//
//                                // Call the receiveFile function here
//                                File imageFile = receiveFile(socket);
//
//                                System.out.print("Image Received");
//
//                                // Preprocess the image and classify it
//                                int predictedClass = classifyImage(updatedModel, imageFile);
//                                System.out.println("Predicted class: " + predictedClass);
//
//
//                                runOnUiThread(() -> bar.setVisibility(View.INVISIBLE));
////
//                                socket.close();
//                                serverSocket.close(); // Close the sockets after sending the response
//
//                            } catch (Exception e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    }).start();
//                }
//            });
//        }




        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            // Hide the progress bar now that we are finished
            final ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            text.setText("The accuracy reach the target " + targetAccuracy + "\n" + "It has taken " + trainNum + " times of training iteration");

            Button inferenceButton = findViewById(R.id.inferenceButton);

            // Make the inference button visible after training
            inferenceButton.setVisibility(View.VISIBLE);

            inferenceButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    bar.setVisibility(View.VISIBLE);
                    // To prevent Main thread blocking
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                ServerSocket serverSocket = new ServerSocket(5001);
                                Socket socket = serverSocket.accept();
                                System.out.println("Connection from " + socket + "!");

                                // Call the receiveFile function here
                                System.out.println("About to receive file...");
                                File imageFile = receiveFile(socket);

                                System.out.println("File received. About to classify image...");
                                int predictedClass = classifyImage(updatedModel, imageFile);

                                System.out.println("Image classified. About to send response...");
// Send the predicted class back to the client

                                socket.close();



                                Socket responseSocket = serverSocket.accept();
                                OutputStream os = responseSocket.getOutputStream();
                                PrintWriter writer = new PrintWriter(os, true);
                                writer.println(predictedClass);

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        bar.setVisibility(View.INVISIBLE);
                                    }
                                });

                                responseSocket.close();
                                serverSocket.close();


                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                    }).start();
                }
            });
        }











































//        protected void onPostExecute(String result) {
//            super.onPostExecute(result);
//            // Hide the progress bar now that we are finished
//            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
//            bar.setVisibility(View.INVISIBLE);
//            text.setText("The accuracy reach the target " + targetAccuracy + "\n" + "It has taken " + trainNum + " times of training iteration");
//
//            Button inferenceButton = findViewById(R.id.inferenceButton);
//
//            // Make the inference button visible after training
//            inferenceButton.setVisibility(View.VISIBLE);
//
//            inferenceButton.setOnClickListener(new View.OnClickListener() {
//                @Override
//                public void onClick(View v) {
//                    bar.setVisibility(View.VISIBLE);
//                    // To prevent Main thread blocking
//                    new Thread(new Runnable() {
//                        @Override
//                        public void run() {
//                            try {
//                                ServerSocket serverSocket = new ServerSocket(5001);
//                                Socket socket = serverSocket.accept();
//                                System.out.println("Connection from " + socket + "!");
//
//                                // Set a timeout on the server socket
//                                socket.setSoTimeout(5000);
//
//                                // Call the receiveFile function here
//                                File imageFile = receiveFile(socket);
//
//                                // Preprocess the image and classify it
//                                int predictedClass = classifyImage(updatedModel, imageFile);
//                                System.out.println("Predicted class: " + predictedClass);
//
//                                // Send the predicted class back to the client
//                                DataOutputStream dos = new DataOutputStream(socket.getOutputStream());
//
//                                //For testing
//                                System.out.println("Sending predicted class: " + predictedClass);
//
//                                try {
//                                    dos.writeInt(predictedClass); // Send the integer directly
//                                }catch (IOException e){
//                                    System.out.println(e);
//                                }
//
//                                bar.setVisibility(View.INVISIBLE);
//
//                                try {
//                                    // This will block until the client reads the data or the timeout occurs
//                                    System.out.println("Stuck here");
//                                    socket.getInputStream().read();
//                                    System.out.println("Stuck here too");
//                                } catch (IOException e) {
//                                    // Handle timeout or other I/O exceptions
//                                    System.out.println("Error reading from client: " + e);
//                                }
//
//                                socket.close();
//                                serverSocket.close(); // Close the sockets after sending the response
//
//                            } catch (Exception e) {
//                                e.printStackTrace();
//                            }
//                        }
//                    }).start();
//                }
//            });
//        }























//        private void receiveFile(Socket socket) {
//            int bytesRead = 0, current = 0;
//
//            try {
//                DataInputStream din = new DataInputStream(socket.getInputStream());
//                DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
//
//                String name = din.readUTF();
//                int fileLength = din.readInt();
//                byte[] byteArray = new byte[fileLength]; // creating byteArray with length same as file length
//                BufferedInputStream bis = new BufferedInputStream(din);
//                String basePath = getCacheDir().getAbsolutePath() + "/server";
//                File file = new File(basePath + "/" + name);
//
//                BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
//                bytesRead = bis.read(byteArray, 0, byteArray.length); // reads bytes of length byteArray from BufferedInputStream & writes into the byteArray, (Offset 0 and length is of byteArray)
//                current = bytesRead;
//
//                // Sometimes only a portion of the file is read, hence to read the remaining portion...
//                do {
//                    // BufferedInputStream is read again into the byteArray, offset is current (which is the amount of bytes read previously) and length is the empty space in the byteArray after current is subtracted from its length
//                    bytesRead = bis.read(byteArray, current, (byteArray.length - current));
//                    if (bytesRead >= 0)
//                        current += bytesRead; // current is updated after the new bytes are read
//                } while (bytesRead > 0);
//
//                bos.write(byteArray, 0, current); // writes bytes from the byteArray into the BufferedOutputStream, offset is 0 and length is current (which is the amount of bytes read into byteArray)
//                bos.close();
//
//                System.out.println("File " + name + " Successfully Downloaded!");
//                dout.writeInt(0); // writeInt is used to reset if any bytes are present in the buffer after the file transfer
//            } catch (IOException ex) {
//                ex.printStackTrace();
//            }
//        }

        // Overloaded Method
//        private void receiveFile(Socket socket) {
//            try {
//                // Get the input stream from the socket
//                InputStream is = socket.getInputStream();
//                BufferedReader reader = new BufferedReader(new InputStreamReader(is));
//
//                // Read the Base64 string from the input stream
//                String imageDataString = reader.readLine();
//
//                // Decode the Base64 string into a byte array
//                byte[] imageData = Base64.decode(imageDataString, Base64.DEFAULT);
//
//                // Write the byte array to a file
//                String basePath = getCacheDir().getAbsolutePath() + "/server";
//                File file = new File(basePath + "/receivedImage.jpg");
//                FileOutputStream fos = new FileOutputStream(file);
//                fos.write(imageData);
//                fos.close();
//
//                System.out.println("Image file received and saved to " + file.getAbsolutePath());
//            } catch (IOException e) {
//                System.out.println("Error receiving image: " + e);
//            }
//        }



//        private void receiveFile(Socket socket) {
//            try {
//                // Get the input stream from the socket
//                InputStream is = socket.getInputStream();
//                BufferedReader reader = new BufferedReader(new InputStreamReader(is));
//
//                // Read the Base64 string from the input stream
//                String imageDataString = reader.readLine();
//
//                // Decode the Base64 string into a byte array
//                byte[] imageData = Base64.decode(imageDataString, Base64.DEFAULT);
//
//                // Write the byte array to a file
//                String basePath = getCacheDir().getAbsolutePath() + "/server";
//                File file = new File(basePath + "/receivedImage.jpg");
//                FileOutputStream fos = new FileOutputStream(file);
//                fos.write(imageData);
//                fos.close();
//
//                System.out.println("Image file received and saved to " + file.getAbsolutePath());
//            } catch (IOException e) {
//                System.out.println("Error receiving image: " + e);
//            }
//        }


        private File receiveFile(Socket socket) {
            try {
                // Get the input stream from the socket
                InputStream is = socket.getInputStream();

                // Read the image data directly into a byte array
                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                int nRead;
                byte[] data = new byte[16384];
                while ((nRead = is.read(data, 0, data.length)) != -1) {
                    buffer.write(data, 0, nRead);
                }

                buffer.flush();
                byte[] imageData = buffer.toByteArray();

                // Write the byte array to a file
                String basePath = getCacheDir().getAbsolutePath() + "/server";
                File file = new File(basePath + "/receivedImage.jpg");
                FileOutputStream fos = new FileOutputStream(file);
                fos.write(imageData);
                fos.close();

                System.out.println("Image file received and saved to " + file.getAbsolutePath());
                return file;
            } catch (IOException e) {
                System.out.println("Error receiving image: " + e);
                return null;
            }
        }

//        private File receiveFile(Socket socket) throws IOException {
//            try {
//                // Get the input stream from the socket
//                InputStream is = socket.getInputStream();
//
//                // Read the image data directly into a byte array
//                ByteArrayOutputStream buffer = new ByteArrayOutputStream();
//                int nRead;
//                byte[] data = new byte[16384];
//                while ((nRead = is.read(data, 0, data.length)) != -1) {
//                    buffer.write(data, 0, nRead);
//                }
//
//                buffer.flush();
//                byte[] imageData = buffer.toByteArray();
//
//                // Write the byte array to a file
//                String basePath = getCacheDir().getAbsolutePath() + "/server";
//                File file = new File(basePath + "/receivedImage.jpg");
//                FileOutputStream fos = new FileOutputStream(file);
//                fos.write(imageData);
//                fos.flush(); // Flush the FileOutputStream
//                fos.close();
//
//                System.out.println("Image file received and saved to " + file.getAbsolutePath());
//                return file;
//            } catch (IOException e) {
//                System.out.println("Error receiving image: " + e);
//                return null;
//            }
//        }









        private int classifyImage(MultiLayerNetwork myNetwork, File imageFile) throws IOException {

            modelBuildAndTrainAndEval model = new modelBuildAndTrainAndEval();

            // Load the image file into a INDArray
            NativeImageLoader loader = new NativeImageLoader(model.numRows, model.numColumns, model.channels);
            INDArray image = loader.asMatrix(imageFile);

            // Preprocess the image
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);

            // Pass the preprocessed image through the model
            INDArray output = myNetwork.output(image);

            // Get the predictions
            int predictedClass = output.argMax(1).getInt(0);

            return predictedClass;
        }


















//// Decryption using Caeser Cipher
//        public void decryptModelFile(File encryptedModelFile) throws Exception {
//            // Read encrypted model data
//            byte[] encryptedData = Files.readAllBytes(encryptedModelFile.toPath());
//
//            // Decrypt each byte using Caesar cipher with negative key
//            byte[] decryptedData = new byte[encryptedData.length];
//            for (int i = 0; i < encryptedData.length; i++) {
//                byte originalByte = encryptedData[i];
//                int shiftedValue = (originalByte - ENCRYPTION_KEY + 256) % 256;  // Handle negative shift values
//                decryptedData[i] = (byte) shiftedValue;
//            }
//
//            Files.write(encryptedModelFile.toPath(), decryptedData);
//        }



//// Decryption using AES-128 ECB mode

//        public void decryptModelFile(File encryptedModelFile) throws Exception {
//            // Read encrypted model data
//            byte[] encryptedData = Files.readAllBytes(encryptedModelFile.toPath());
//
//            // Assuming KEY_VALUE is defined as a byte array with a length of 16 bytes
//            byte[] key = KEY_VALUE.length == 16 ?
//                    new SecretKeySpec(KEY_VALUE, "AES").getEncoded() : null;
//            if (key == null) {
//                throw new IllegalArgumentException("Static key size must be 16 bytes for AES-128");
//            }
//
//            // Create an AES cipher object
//            Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
//            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"));
//
//            // Decrypt the model data
//            byte[] decryptedData = cipher.doFinal(encryptedData);
//
//            // Write decrypted data to file (optional)
//            // You can write the decrypted data to a new file or use it within your application
//             Files.write(encryptedModelFile.toPath(), decryptedData);
//        }



//// Decryption using AES-128 CBC mode

//
//        public void decryptModelFile(File encryptedModelFile) throws Exception {
//            // Read encrypted data (including IV) from file
//            byte[] encryptedDataWithIv = Files.readAllBytes(encryptedModelFile.toPath());
//
//            // Assuming KEY_VALUE is defined as a byte array with a length of 16 bytes
//            byte[] key = KEY_VALUE.length == 16 ?
//                    new SecretKeySpec(KEY_VALUE, "AES").getEncoded() : null;
//            if (key == null) {
//                throw new IllegalArgumentException("Static key size must be 16 bytes for AES-128");
//            }
//
//            // Extract IV from the beginning of encrypted data
//            byte[] iv = new byte[16];
//            System.arraycopy(encryptedDataWithIv, 0, iv, 0, iv.length);
//
//            // Extract encrypted data after the IV
//            byte[] encryptedData = new byte[encryptedDataWithIv.length - iv.length];
//            System.arraycopy(encryptedDataWithIv, iv.length, encryptedData, 0, encryptedData.length);
//
//            // Create an AES cipher object
//            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
//            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"), new IvParameterSpec(iv));
//
//            // Decrypt the model data
//            byte[] decryptedData = cipher.doFinal(encryptedData);
//
//            // Write decrypted data to file (optional)
//            // You can write the decrypted data to a new file or use it within your application
//            Files.write(encryptedModelFile.toPath(), decryptedData);
//        }




    // Decryption using AES-256 GCM mode
        public void decryptModelFile(File modelFile) throws Exception {
            // Read the encrypted data from the file
            byte[] encryptedDataWithIvAndTag = Files.readAllBytes(modelFile.toPath());

            // Split the encrypted data into IV, cipher text, and tag
            if (encryptedDataWithIvAndTag.length < 12 + 16) {
                throw new IllegalArgumentException("Invalid encrypted data format");
            }

            byte[] iv = new byte[12];
            System.arraycopy(encryptedDataWithIvAndTag, 0, iv, 0, iv.length);

            byte[] cipherTextWithTag = new byte[encryptedDataWithIvAndTag.length - iv.length];
            System.arraycopy(encryptedDataWithIvAndTag, iv.length, cipherTextWithTag, 0, cipherTextWithTag.length);

            // Generate a SecretKey from the static key (not recommended in production)
            // Assuming KEY_VALUE is defined as a byte array with a length of 32 bytes (required for AES-256)
            byte[] keyBytes = KEY_VALUE.length == 32 ? KEY_VALUE : null;

            if (keyBytes == null) {
                throw new IllegalArgumentException("Static key size must be 32 bytes for AES-256 with GCM");
            }

            // Create a SecretKeySpec object
            SecretKeySpec key = new SecretKeySpec(keyBytes, "AES");

            // Create a GCM Cipher object
            Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");

            // Use GCMParameterSpec with the IV
            GCMParameterSpec spec = new GCMParameterSpec(128, iv); // Tag length in bits (typically 128 for GCM)

            // Initialize cipher for decryption mode with key and IV spec
            cipher.init(Cipher.DECRYPT_MODE, key, spec);

            // Decrypt the cipher text
            byte[] decryptedData = cipher.doFinal(cipherTextWithTag);

            // Write the decrypted data back to the model file
            Files.write(modelFile.toPath(), decryptedData);
        }




        // The function for the weight averaging
        /*
        private void weightAveraging(MultiLayerNetwork modelLoad_cID_1,MultiLayerNetwork modelLoad_cID_2) throws IOException {
            INDArray meanCoefficient;
            meanCoefficient = modelLoad_cID_1.params().addi(modelLoad_cID_2.params()).divi(2);
            INDArray meanUpdaterState;
            meanUpdaterState = modelLoad_cID_1.updaterState().addi(modelLoad_cID_2.updaterState()).divi(2);
            MultiLayerNetwork updatedModel = new MultiLayerNetwork(modelLoad_cID_1.getLayerWiseConfigurations(), meanCoefficient);
            updatedModel.getUpdater().setStateViewArray(updatedModel, meanUpdaterState, false);

            String basePath = getCacheDir().getAbsolutePath() + "/server";
            ModelSerializer.writeModel(updatedModel, new File(basePath + "/updatedModel.zip"), true);
            System.out.println("Model has been updated in the server");
        }
        */



        private void weightAveraging(MultiLayerNetwork modelLoad_cID_1, MultiLayerNetwork modelLoad_cID_2) throws IOException {

            // Outlier detection threshold (adjust as needed)
            double outlierThreshold = 3.0; // Standard deviation multiplier

            // Access model parameters for averaging
            INDArray params_cID_1 = modelLoad_cID_1.params();
            INDArray params_cID_2 = modelLoad_cID_2.params();

            // Calculate difference between parameters (assuming same shape)
            INDArray paramDifference = params_cID_1.sub(params_cID_2);

            // Calculate mean and standard deviation of the parameter difference
            double meanDifference = Nd4j.mean(paramDifference).getDouble(0);
            double stdDevDifference = Nd4j.std(paramDifference).getDouble(0);

            // Identify potential outliers based on difference from mean
            List<Integer> outlierIndices = new ArrayList<>();
            for (int i = 0; i < paramDifference.length(); i++) {
                double value = paramDifference.getDouble(i);
                double distanceFromMean = Math.abs(value - meanDifference);
                if (distanceFromMean > outlierThreshold * stdDevDifference) {
                    outlierIndices.add(i);
                }
            }

            // Handle outliers (consider filtering or clipping)
            INDArray averagedParams;
            if (outlierIndices.isEmpty()) {
                // No outliers detected, use simple averaging
                averagedParams = params_cID_1.addi(params_cID_2).divi(2.0);
            } else {
                // Outliers detected, consider two options:
                // Option 1: Filter outliers (potentially discarding valuable information)
                // averagedParams = handleOutliersWithFiltering(params_cID_1, params_cID_2, outlierIndices);

                // Option 2: Clip outliers to a threshold (potentially less discarding)
                // ... outlier detection logic to populate outlierIndices ...
                averagedParams = handleOutliersWithClipping(params_cID_1, params_cID_2, outlierThreshold, meanDifference, stdDevDifference, outlierIndices);

            }

            // Rest of your code for creating and saving the updated model:
            INDArray meanUpdaterState = modelLoad_cID_1.updaterState().addi(modelLoad_cID_2.updaterState()).divi(2);
            MultiLayerNetwork updatedModel = new MultiLayerNetwork(modelLoad_cID_1.getLayerWiseConfigurations(), averagedParams);
            updatedModel.getUpdater().setStateViewArray(updatedModel, meanUpdaterState, false);

            String basePath = getCacheDir().getAbsolutePath() + "/server";
            ModelSerializer.writeModel(updatedModel, new File(basePath + "/updatedModel.zip"), true);
            System.out.println("Model has been updated in the server");
        }


        private INDArray handleOutliersWithClipping(INDArray params_cID_1, INDArray params_cID_2, double outlierThreshold, double meanDifference, double stdDevDifference, List<Integer> outlierIndices) {
            INDArray averagedParams = Nd4j.create(params_cID_1.shape());

            for (int i = 0; i < params_cID_1.length(); i++) {
                double value1 = params_cID_1.getDouble(i);
                double value2 = params_cID_2.getDouble(i);
                if (!outlierIndices.contains(i)) {
                    // Not an outlier, use simple averaging
                    averagedParams.putScalar(i, (value1 + value2) / 2.0);
                } else {
                    // Outlier detected, clip value to threshold around mean
                    double clippedValue = Math.signum(value1 - meanDifference) * outlierThreshold * stdDevDifference + meanDifference;
                    averagedParams.putScalar(i, clippedValue);
                }
            }
            return averagedParams;
        }


    }

    // The class for building the model and evaluate the model
    // It has two methods. One is to generate the initial model. The other is to do the aggregation model evaluation.
    private class modelBuildAndTrainAndEval {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);

        private void modelGenrated() throws IOException {

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

            String basePath = getCacheDir().getAbsolutePath() + "/server";
            ModelSerializer.writeModel(myNetwork, new File(basePath + "/updatedModel.zip"), false);
            System.out.println("The initial model has been generated!");
        }


        private Evaluation modelEval(MultiLayerNetwork myNetwork) throws IOException, InterruptedException {

            // vectorization of test data
            String basePath = getCacheDir().getAbsolutePath() + "/server";
            File testData = new File(basePath + "/mnist_test/mnist_test");
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
            testRR.initialize(testSplit);
            DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            mnistTest.setPreProcessor(scaler);

            Log.d("evaluate model", "Evaluate model....");
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





    // The class for multithreads of receiving service
    class ServiceReceive implements Runnable {
        private DataInputStream din;
        private DataOutputStream dout;
        public Socket socket;

        public ServiceReceive(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                receiveFile();
                Thread.sleep(10000000);
            } catch (IOException | InterruptedException e) {
                try {
                    System.out.println("interrupt the thread... ");
                    dout.writeUTF("FINISH");
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        }

        // receive the files from the clients
        private void receiveFile() {
            int bytesRead = 0, current = 0;

            try {
                String name = din.readUTF();
                int fileLength = din.readInt();
                byte[] byteArray = new byte[fileLength];                    //creating byteArray with length same as file length
                BufferedInputStream bis = new BufferedInputStream(din);
                String basePath = getCacheDir().getAbsolutePath() + "/server";
                File file = new File(basePath + "/" + name);
                //fileFoundFlag is a Flag which denotes the file is present or absent from the Server directory, is present int 0 is sent, else 1
                int fileFoundFlag = din.readInt();
                //System.out.println(fileFoundFlag);
                if (fileFoundFlag == 1)
                    return;
                BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
                bytesRead = bis.read(byteArray, 0, byteArray.length);            //reads bytes of length byteArray from BufferedInputStream & writes into the byteArray, (Offset 0 and length is of byteArray)
                current = bytesRead;
                //Sometimes only a portion of the file is read, hence to read the remaining portion...
                do {
                    //BufferedInputStream is read again into the byteArray, offset is current (which is the amount of bytes read previously) and length is the empty space in the byteArray after current is subtracted from its length
                    bytesRead = bis.read(byteArray, current, (byteArray.length - current));
                    if (bytesRead >= 0)
                        current += bytesRead;                    //current is updated after the new bytes are read
                } while (bytesRead > 0);
                bos.write(byteArray, 0, current);                //writes bytes from the byteArray into the BufferedOutputStream, offset is 0 and length is current (which is the amount of bytes read into byteArray)
                bos.close();
                System.out.println("        File " + " Successfully Downloaded!");
                dout.writeInt(0);                        //writeInt is used to reset if any bytes are present in the buffer after the file transfer
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    // The class for multithreads of sending service
    class ServiceSend implements Runnable {
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;

        public ServiceSend(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("model sending ...");
                String basePath = getCacheDir().getAbsolutePath() + "/server";
                File initModel = new File(basePath + "/updatedModel.zip");
                sendFile(initModel);
                System.out.println("model sent");
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

        // send the files to clients
        public void sendFile(File file) {
            try {
                byte[] byteArray = new byte[(int) file.length()];                    //creating byteArray with length same as file length
                dout.writeInt(byteArray.length);
                BufferedInputStream bis = new BufferedInputStream(new FileInputStream(file));
                //Writing int 0 as a Flag which denotes the file is present in the Server directory, if file was absent, FileNotFound exception will be thrown and int 1 will be written
                dout.writeInt(0);
                BufferedOutputStream bos = new BufferedOutputStream(dout);

                int count;
                while ((count = bis.read(byteArray)) != -1) {            //reads bytes of byteArray length from the BufferedInputStream into byteArray
                    bos.write(byteArray, 0, count);                    //writes bytes from byteArray into the BufferedOutputStream (0 is the offset and count is the length)
                }
                bos.flush();
                bis.close();
                din.readInt();//readInt is used to reset if any bytes are present in the buffer after the file transfer
            } catch (FileNotFoundException ex) {
                System.out.println("File Not Found!");
                try {
                    //Writing int 1 as a Flag which denotes the file is absent from the Server directory, if file was present int 0 would be written
                    dout.writeInt(1);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }

        }
    }

    // The class for multithreads of sending continuing signal to clients so that clients can continue train the model
    class ContinueSignalSend implements Runnable {
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;

        public ContinueSignalSend(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("Send the signal to inform client of continuing training or not...");
                dout.writeInt(1);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }

    // The class for multithreads of sending stop signal to clients so that clients do not continue train the model
    class stopSignalSend implements Runnable {
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;

        public stopSignalSend(Socket socket) {
            this.socket = socket;
        }

        @Override
        public void run() {
            try {
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("Send the signal to inform client of continuing training or not...");
                dout.writeInt(0);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }

}

