package com.example.mnist;

import android.Manifest;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Debug;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
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
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Random;

import au.com.bytecode.opencsv.CSVWriter;

import java.nio.file.Files;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;



//import android.support.v7.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {
    //private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";

    //String basePath = getCacheDir().getAbsolutePath() + "/mnist";
    //IID Distribution
    private static final String mnistTrainUrl = "https://github.com/AyushSharma5583/Federated-Learning/blob/main/dataset/mnist_client1_iid.tar.gz?raw=true";


    //Non-IID Distribution
    //private static final String mnistTrainUrl = "https://github.com/AyushSharma5583/Federated-Learning/blob/main/dataset/mnist_client1_non_iid.tar.gz?raw=true";
    private static final String clientID = "1";
    // the I/O stream for sending and receiving the model
    private static DataInputStream din;
    private static DataOutputStream dout;
    double trainTime;
    TextView textTime;
    TextView textAccuracy;
    static String serverIP = "100.74.134.184";
    String trainDataset = "client1_mnist_iid_batch";


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



    private static final int IMAGE_PICK_CODE = 1;


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


        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("Started training...");
        }

        // This is our main background thread for training the model and uploading the model
        @Override
        protected String doInBackground(String... params) {
            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            try {

                // download the dataset from the Internet
                if (!new File(basePath + "/mnist_client1_iid").exists()) {
                    Log.d("Data download", "Data downloaded from " + mnistTrainUrl);
                    File modelDir = new File(basePath + "/mnist_client1_iid");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTrainUrl, basePath)) {
                        DataUtilities.extractTarGz(basePath+"/mnist_client1_iid.tar.gz", basePath + "/mnist_client1_iid");
                    }
                }
                // the beginning timestamp
                double beginTime = System.nanoTime();

                // write the training time and current absolute time to csv of each round
                File file = new File(basePath + "/"+ trainDataset + ".csv");
                FileWriter output = null;
                try {
                    output = new FileWriter(file);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                CSVWriter write = new CSVWriter(output);
                // Header column value
                String[] header = { "ID", "Training Time", "Current Time" };
                int num = 0;
                write.writeNext(header);

                // load the training dataset
                modelTrain model = new modelTrain();
                DataSetIterator mnistTrain = model.loadTrainData();

                // The training process
                // First it will receive the model from the server and then load the model to train it using its own dataset
                // After that, send the trained model back to the server and waiting for the new model from the server
                while(true){
                    receiveModel(MainActivity.this);
                    String updatedModelPath = basePath + "/updatedModel.zip";
                    MultiLayerNetwork modelLoad = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath);
                    trainTime = model.modelTrain(modelLoad,mnistTrain);

                    // Encryption
                    encryptModelFile(new File(basePath + "/localModel_cID_" + clientID + ".zip"));

                    sendModel(MainActivity.this);
                    num = num + 1;
                    if (receiveSignal() == 1){
                        Thread.sleep(5000);
                        double currentTime = (System.nanoTime()-beginTime)/1000000000;
                        String[] data={String.valueOf(num),String.valueOf(trainTime), String.valueOf(currentTime)};
                        write.writeNext(data);
                        continue;
                    }
                    else{
                        double currentTime = (System.nanoTime()-beginTime)/1000000000;
                        String[] data={String.valueOf(num),String.valueOf(trainTime), String.valueOf(currentTime)};
                        write.writeNext(data);
                        write.close();


                        break;
                    }
                }

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


        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            // Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("The time for training is " + trainTime + "s");
            requestReadMediaImagesPermission();
            Button selectImageButton = findViewById(R.id.selectImageButton); // Initialize button after training
            selectImageButton.setVisibility(View.VISIBLE); // Make button visible
            selectImageButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    // Open image selection dialog or gallery
                    Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(intent, IMAGE_PICK_CODE); // Request code for image selection
                }
            });
        }

    }


    private static final int REQUEST_CODE_READ_MEDIA_IMAGES = 1;
    private void requestReadMediaImagesPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_CODE_READ_MEDIA_IMAGES);
        } else {
            // Permission already granted, proceed with image access logic
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_READ_MEDIA_IMAGES) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, proceed with image access logic
            } else {
                // Permission denied, inform user or handle the situation accordingly
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == IMAGE_PICK_CODE && resultCode == RESULT_OK) {
            String imagePath = data.getDataString(); // Get image path from intent

            Log.d("ImagePath", imagePath);

            // Start a new AsyncTask to send the image in the background
            new SendImageTask(imagePath).execute();
        }
    }


    private class SendImageTask extends AsyncTask<Void, Void, Void> {
        private String imagePath;
        public SendImageTask(String imagePath) {
            this.imagePath = imagePath;
        }


        protected Void doInBackground(Void... voids) {

            try {
                Socket socket = new Socket(serverIP, 5001);

                // Check if imagePath is a content URI
                if (imagePath.startsWith("content://")) {
                    // Use ContentResolver to open the image data
                    ContentResolver resolver = getContentResolver();
                    InputStream inputStream = resolver.openInputStream(Uri.parse(imagePath));
                    byte[] imageData = new byte[inputStream.available()];
                    inputStream.read(imageData);
                    inputStream.close();

                    // Send image data over the socket
                    OutputStream os = socket.getOutputStream();
                    os.write(imageData);
                    os.close();

                } else {
                    // Existing code for handling regular file paths
                    FileInputStream imageFile = new FileInputStream(imagePath);
                    byte[] imageData = new byte[imageFile.available()];
                    imageFile.read(imageData);
                    imageFile.close();

                    // Send image data over the socket
                    OutputStream os = socket.getOutputStream();
                    os.write(imageData);
                    os.close();

                }

                socket.close();

                Socket responseSocket = new Socket(serverIP, 5001);

                InputStream is = responseSocket.getInputStream();
                BufferedReader reader = new BufferedReader(new InputStreamReader(is));
                int predictedClass = Integer.parseInt(reader.readLine());
                System.out.println("Predicted class: " + predictedClass);

                responseSocket.close();

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        TextView textView = findViewById(R.id.predictedClassTextView);
                        textView.setText("Predicted class: " + predictedClass);
                        textView.setVisibility(View.VISIBLE);
                    }
                });



            } catch (FileNotFoundException e) {
                System.out.println("Error: Image file not found at '" + imagePath + "'.");
            } catch (Exception e) {
                System.out.println("Error sending image: " + e);
            }
            return null;
        }


    }





    // The class for train the model
    // It has two methods. One is load the training data and the other is train the model.
    private  class modelTrain {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);
        int numEpochs = 1; // number of epochs to perform
        int numBatch = 25;

        private DataSetIterator loadTrainData() throws IOException, InterruptedException {
            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            // vectorization of train data
            File trainData = new File(basePath + "/mnist_client1_iid");
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



        private double modelTrain(MultiLayerNetwork myNetwork, DataSetIterator mnistTrain) throws Exception {
            String basePath = getCacheDir().getAbsolutePath() + "/mnist";
            Log.d("train model", "Train model....");

            double startTime = System.nanoTime();
            for (int l = 0; l < numBatch; l++) {
                DataSet ds = mnistTrain.next();
                myNetwork.fit(ds);
            }
//            myNetwork.fit(mnistTrain, numEpochs);
            trainTime = (System.nanoTime()-startTime)/1000000000;
            System.out.println("The time for training is "+ trainTime + "s");
            ModelSerializer.writeModel(myNetwork,  new File(basePath+"/localModel_cID_" + clientID +".zip" ), true);

            return trainTime;
        }

    }



//// Model Encryption using Caeser Cipher
//    public void encryptModelFile(File modelFile) throws Exception {
//        // Read model file content
//        byte[] modelData = Files.readAllBytes(modelFile.toPath());
//
//        // Encrypt each byte using Caesar cipher
//        byte[] encryptedData = new byte[modelData.length];
//        for (int i = 0; i < modelData.length; i++) {
//            byte originalByte = modelData[i];
//            int shiftedValue = (originalByte + ENCRYPTION_KEY) % 256;  // Wrap around for non-alphanumeric characters
//            encryptedData[i] = (byte) shiftedValue;
//        }
//
//        // Overwrite model file with encrypted data
//        Files.write(modelFile.toPath(), encryptedData);
//    }




//// Model Encryption using AES-128 ECB mode
//    public void encryptModelFile(File modelFile) throws Exception {
//        // Read model file content
//        byte[] modelData = Files.readAllBytes(modelFile.toPath());
//
//        // Generate a SecretKey from the static key (not recommended in production)
//        // Assuming KEY_VALUE is defined as a byte array with a length of 16 bytes
//        byte[] key = KEY_VALUE.length == 16 ?
//                new SecretKeySpec(KEY_VALUE, "AES").getEncoded() : null; // Adjust key size as needed
//        if (key == null) {
//            throw new IllegalArgumentException("Static key size must be 16 bytes for AES-128");
//        }
//
//        // Create an AES cipher object
//        Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
//        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"));
//
//        // Encrypt the model data
//        byte[] encryptedData = cipher.doFinal(modelData);
//
//        // Overwrite model file with encrypted data
//        Files.write(modelFile.toPath(), encryptedData);
//    }




////     Model Encryption using AES-128 CBC mode
//    public void encryptModelFile(File modelFile) throws Exception {
//        // Read model file content
//        byte[] modelData = Files.readAllBytes(modelFile.toPath());
//
//        // Generate a SecretKey from the static key (not recommended in production)
//        // Assuming KEY_VALUE is defined as a byte array with a length of 16 bytes
//        byte[] key = KEY_VALUE.length == 16 ?
//                new SecretKeySpec(KEY_VALUE, "AES").getEncoded() : null;
//        if (key == null) {
//            throw new IllegalArgumentException("Static key size must be 16 bytes for AES-128");
//        }
//
//        // Generate a random Initialization Vector (IV)
//        byte[] iv = new byte[16]; // Size should match cipher block size (16 bytes for AES)
//        new SecureRandom().nextBytes(iv);
//
//        // Create an AES cipher object
//        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
//        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"), new IvParameterSpec(iv));
//
//        // Encrypt the model data
//        byte[] encryptedData = cipher.doFinal(modelData);
//
//        // Combine IV and encrypted data (prepend IV to encrypted data)
//        byte[] encryptedDataWithIv = new byte[iv.length + encryptedData.length];
//        System.arraycopy(iv, 0, encryptedDataWithIv, 0, iv.length);
//        System.arraycopy(encryptedData, 0, encryptedDataWithIv, iv.length, encryptedData.length);
//
//        // Write encrypted data (including IV) to file
//        Files.write(modelFile.toPath(), encryptedDataWithIv);
//    }



    // Model Encryption using AES-256 GCM mode
    public void encryptModelFile(File modelFile) throws Exception {
        // Read model file content
        byte[] modelData = Files.readAllBytes(modelFile.toPath());

        // Generate a SecretKey from the static key (not recommended in production)
        // Assuming KEY_VALUE is defined as a byte array with a length of 32 bytes (required for AES-256)
        byte[] keyBytes = KEY_VALUE.length == 32 ? KEY_VALUE : null;

        if (keyBytes == null) {
            throw new IllegalArgumentException("Static key size must be 32 bytes for AES-256 with GCM");
        }

        // Create a SecretKeySpec object
        SecretKeySpec key = new SecretKeySpec(keyBytes, "AES");

        // Generate a random initialization vector (IV)
        byte[] iv = new byte[12]; // Size typically 12 bytes for GCM
        new SecureRandom().nextBytes(iv);

        // Create a GCM Cipher object
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");

        // Use GCMParameterSpec with the IV
        GCMParameterSpec spec = new GCMParameterSpec(128, iv); // Tag length in bits (typically 128 for GCM)

        // Initialize cipher for encryption mode with key and IV spec
        cipher.init(Cipher.ENCRYPT_MODE, key, spec);

        // Encrypt the model data
        byte[] cipherText = cipher.doFinal(modelData);

        // Combine IV, encrypted data, and tag (prepend IV)
        byte[] encryptedDataWithIvAndTag = new byte[iv.length + cipherText.length];
        System.arraycopy(iv, 0, encryptedDataWithIvAndTag, 0, iv.length);
        System.arraycopy(cipherText, 0, encryptedDataWithIvAndTag, iv.length, cipherText.length);

        // Write the encrypted data back to the model file
        Files.write(modelFile.toPath(), encryptedDataWithIvAndTag);
    }






    // This is function for receiving the model from the server
    private static void receiveModel(Context context) throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());
        System.out.println("Receiving model from server...");


        receiveFile(context);
        System.out.println("Model Received!");

        System.out.println("Closing socket.");
        socket.close();
    }

    // This is function for sending the model to the server
    private static void sendModel(Context context) throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());

        String basePath = context.getCacheDir().getAbsolutePath() + "/mnist";
        File fileSent = new File(basePath+"/localModel_cID_" + clientID +".zip" );
        System.out.println("Trained Model sending to server...");
        sendFile_from_client(fileSent);			//the file if present, is sent over the network
        System.out.println("Trained Model sent!");
        System.out.println(din.readUTF());
        System.out.println("Closing socket and terminating program.");;
        socket.close();
    }

    // This is the function to know if the server asks the clients to stop training as the model has reached the target accuracy
    private static int receiveSignal() throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());
        System.out.println("Receiving signal from server to decide if continuing training...");
        int signal = din.readInt();

        socket.close();
        return signal ;
    }

    // The function for sending files using sockets
    private static void sendFile_from_client(File file) {
        try {
            dout.writeUTF(file.getName());
            //creating byteArray with length same as file length
            byte[] byteArray = new byte[(int) file.length()];
            dout.writeInt(byteArray.length);
            BufferedInputStream bis = new BufferedInputStream (new FileInputStream(file));
            //Writing int 0 as a Flag which denotes the file is present in the Server directory, if file was absent, FileNotFound exception will be thrown and int 1 will be written
            dout.writeInt(0);
            BufferedOutputStream bos = new BufferedOutputStream(dout);
            int count;
            while((count = bis.read(byteArray)) != -1) {			//reads bytes of byteArray length from the BufferedInputStream into byteArray
                //writes bytes from byteArray into the BufferedOutputStream (0 is the offset and count is the length)
                bos.write(byteArray, 0, count);
            }
            bos.flush();
            bis.close();
            //readInt is used to reset if any bytes are present in the buffer after the file transfer
            din.readInt();
        }
        catch(FileNotFoundException ex) {
            System.out.println("File "  + " Not Found! \n        Please Check the input and try again.\n\n        ");
            try {
                //Writing int 1 as a Flag which denotes the file is absent from the Server directory, if file was present int 0 would be written
                dout.writeInt(1);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        catch(IOException ex) {
            ex.printStackTrace();
        }
    }

    // The function for receiving files using sockets
    private static void receiveFile(Context context) {
        int bytesRead = 0, current = 0;
        try {
            int fileLength = din.readInt();
            //creating byteArray with length same as file length
            byte[] byteArray = new byte[fileLength];
            BufferedInputStream bis = new BufferedInputStream(din);

            String basePath = context.getCacheDir().getAbsolutePath() + "/mnist";
            File file = new File(basePath+"/updatedModel.zip");
            //fileFoundFlag is a Flag which denotes the file is present or absent from the Server directory, is present int 0 is sent, else 1
            int fileFoundFlag = din.readInt();
            if(fileFoundFlag == 1)
                return;
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
            //reads bytes of length byteArray from BufferedInputStream & writes into the byteArray, (Offset 0 and length is of byteArray)
            bytesRead = bis.read(byteArray, 0, byteArray.length);
            current = bytesRead;
            //Sometimes only a portion of the file is read, hence to read the remaining portion...
            do {
                //BufferedInputStream is read again into the byteArray, offset is current (which is the amount of bytes read previously) and length is the empty space in the byteArray after current is subtracted from its length
                bytesRead = bis.read(byteArray, current, (byteArray.length - current));

                if(bytesRead >= 0)
                    current += bytesRead;					//current is updated after the new bytes are read
            } while(bytesRead > 0);
            //writes bytes from the byteArray into the BufferedOutputStream, offset is 0 and length is current (which is the amount of bytes read into byteArray)
            bos.write(byteArray, 0, current);
            bos.close();
            System.out.println("Model Successfully Downloaded!" );
            //writeInt is used to reset if any bytes are present in the buffer after the file transfer
            dout.writeInt(0);
        }
        catch(IOException ex) {
            ex.printStackTrace();
        }
    }
}