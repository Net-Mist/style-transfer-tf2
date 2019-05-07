package com.styletransfer;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Logger;

public class MainActivity extends AppCompatActivity {
    static final int REQUEST_IMAGE_CAPTURE = 1;

    /**
     * Optional GPU delegate for accleration.
     */
    private GpuDelegate gpuDelegate = null;
    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;
    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;
    /**
     * A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
     */
    protected ByteBuffer imgData = null;
    private float[][] outputData = null;

    private int networkX = 658;
    private int networkY = 493;
    /** Preallocated buffers for storing image data in. */
    private final int[] intValues = new int[networkX * networkY];

    String currentPhotoPath;

    private final static Logger LOGGER = Logger.getLogger(MainActivity.class.getName());


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        tfliteOptions.setNumThreads(1);

        try {
            tfliteModel = loadModelFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        tflite = new Interpreter(tfliteModel, tfliteOptions);

        imgData = ByteBuffer.allocateDirect(networkY * networkX * 3 * 4);
        imgData.order(ByteOrder.nativeOrder());


    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Called when the user taps the Send button
     */
    public void takePicture(View view) {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.example.android.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            ImageView imageView = (ImageView) findViewById(R.id.imageView);

            // to get small image...
//            Bundle extras = data.getExtras();
//            Bitmap imageBitmap = (Bitmap) extras.get("data");
//            imageView.setImageBitmap(imageBitmap);

            // Get the dimensions of the View
            int targetW = 640; //imageView.getWidth();
            int targetH = 480; //imageView.getHeight();

            // Get the dimensions of the bitmap
            BitmapFactory.Options bmOptions = new BitmapFactory.Options();
            bmOptions.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(currentPhotoPath, bmOptions);
            int photoW = bmOptions.outWidth;
            int photoH = bmOptions.outHeight;

            // Determine how much to scale down the image
            int scaleFactor = Math.min(photoW / targetW, photoH / targetH);

            // Decode the image file into a Bitmap sized to fill the View
            bmOptions.inJustDecodeBounds = false;
            bmOptions.inSampleSize = scaleFactor;
            bmOptions.inPurgeable = true;

            Bitmap bitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions);

            // apply tflite
            imgData.rewind();
            LOGGER.info(String.valueOf(bitmap.getWidth()));
            LOGGER.info(String.valueOf(bitmap.getHeight()));
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            // Convert the image to floating point.
            int pixel = 0;
            for (int i = 0; i < 640; ++i) {
                for (int j = 0; j < 480; ++j) {
                    final int val = intValues[pixel++];
                    addPixelValue(val);
                }
            }
            tflite.run(imgData, outputData);



            imageView.setImageBitmap(bitmap);


        }
    }

    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) - 127.5f -1.f);
        imgData.putFloat(((pixelValue >> 8) & 0xFF) - 127.5f -1.f);
        imgData.putFloat((pixelValue & 0xFF) - 127.5f -1.f);
    }


    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }


}
