package com.cursoandroid.handlandmark;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraInfoUnavailableException;
import androidx.camera.core.CameraProvider;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.cursoandroid.handlandmark.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.zip.Inflater;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import ai.onnxruntime.*;

public class MainActivity extends AppCompatActivity {

    private BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task");
    private BaseOptions baseOptions = baseOptionsBuilder.build();
    private HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.55f)
            .setMinHandPresenceConfidence(0.55f)
            .setRunningMode(RunningMode.IMAGE)
            .build();
    private Button button;
    private TextView textView;
    private ActivityResultLauncher<PickVisualMediaRequest> mpImage;
    private ActivityMainBinding viewBinding;
    private PreviewView previewView;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        HandLandmarker handLandmarker = HandLandmarker.createFromOptions(this, options);
        button = findViewById(R.id.button);
        textView = findViewById(R.id.textView);



        Camera camera =
        /*
            if (uri != null) {
                Bitmap bitmap = uriParaBitmap(uri);
                MPImage mpImg = new BitmapImageBuilder(bitmap).build();
                HandLandmarkerResult resultado = handLandmarker.detect(mpImg);
                List<Float> lista = new ArrayList<>();
                if (!resultado.landmarks().isEmpty()) {
                    for (List<NormalizedLandmark> l : resultado.landmarks()) {
                        for (NormalizedLandmark landmark : l) {
                            lista.add(landmark.x());
                            lista.add(landmark.y());
                        }
                    }
                }
                float[] vet = new float[lista.size()];
                int p = 0;
                for (float i : lista) {
                    vet[p++] = i;
                }

                try {
                    OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
                    OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                    ByteBuffer modelBuffer = loadModelFile("modelo_rf.onnx");
                    OrtSession session = ortEnvironment.createSession(modelBuffer, sessionOptions); //[rotulo, propabilidade]
                    OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(vet), new long[]{1, 42});
                    OrtSession.Result result = session.run(Collections.singletonMap("input", tensor));



                    String[] saida = (String[]) result.get(0).getValue();



                    textView.setText(saida[0]);




                    tensor.close();
                    result.close();
                    session.close();

                }
        });*/

    }

    private Bitmap uriParaBitmap(Uri uri) {
        try {
            Bitmap bitmap;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
                bitmap = ImageDecoder.decodeBitmap(source);
            } else {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            }
            if (bitmap != null && bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            }
            return bitmap;
        } catch (IOException e) {
            Log.e("ImageError", "Erro ao converter URI para Bitmap", e);
            return null;
        }
    }

    private ByteBuffer loadModelFile(String caminho) throws IOException {
        InputStream inputStream = getAssets().open(caminho);

        byte[] modelBytes = new byte[inputStream.available()];
        inputStream.read(modelBytes);
        inputStream.close();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelBytes.length);
        byteBuffer.put(modelBytes);
        byteBuffer.rewind();

        return byteBuffer;
    }
    private void bindPreview(@NonNull ProcessCameraProvider processCameraProvider){
        Preview preview = new Preview.Builder()
                .build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        Camera camera = (Camera) processCameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview);
    }

}
