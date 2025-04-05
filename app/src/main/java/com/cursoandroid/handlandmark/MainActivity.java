package com.cursoandroid.handlandmark;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import androidx.viewbinding.ViewBinding;

import com.cursoandroid.handlandmark.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;

import ai.onnxruntime.*;

public class MainActivity extends AppCompatActivity {

    private ImageButton  flash, virarCamera;
    private TextView textView;
    private PreviewView previewView;
    private int cameraFacing = CameraSelector.LENS_FACING_BACK;
    private OrtEnvironment ortEnvironment;
    private OrtSession.SessionOptions sessionOptions;
    private ByteBuffer modelBuffer;
    private ImageCapture imageCapture;
    private BaseOptions.Builder baseOptionsBuilder = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task");
    private BaseOptions baseOptions = baseOptionsBuilder.build();
    private HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinHandDetectionConfidence(0.55f)
            .setMinHandPresenceConfidence(0.55f)
            .setRunningMode(RunningMode.IMAGE)
            .build();
    private Handler handler;
    private Runnable runnable;

    private final ActivityResultLauncher<String> activityResultLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(),
            new ActivityResultCallback<Boolean>() {
                @Override
                public void onActivityResult(Boolean granted) {
                    if (granted) {
                        startCamera(cameraFacing);
                    }
                }
            });
    private HandLandmarker handLandmark;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        flash = findViewById(R.id.flash);
        virarCamera = findViewById(R.id.virar_camera);
        textView = findViewById(R.id.textView);
        handLandmark = HandLandmarker.createFromOptions(this, options);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            activityResultLauncher.launch(Manifest.permission.CAMERA);
        } else {
            startCamera(cameraFacing);
        }

        virarCamera.setOnClickListener(v -> {
            cameraFacing = (cameraFacing == CameraSelector.LENS_FACING_BACK) ?
                    CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
            startCamera(cameraFacing);
        });

        try {
            ortEnvironment = OrtEnvironment.getEnvironment();
            sessionOptions = new OrtSession.SessionOptions();
            modelBuffer = loadModelFile("modelo_gestos.onnx");
        } catch (IOException e) {
            Log.e("ONNX", "Erro ao carregar o modelo", e);
            return;
        }

        handler = new Handler();
        runnable = new Runnable() {
            @Override
            public void run() {
                capturarFoto();
                handler.postDelayed(runnable, 50);
            }
        };
        runnable.run();

        flash.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            }
        });
    }

    private void capturarFoto() {
        if (imageCapture != null) {
            imageCapture.takePicture(ContextCompat.getMainExecutor(this), new ImageCapture.OnImageCapturedCallback() {
                @Override
                public void onCaptureSuccess(@NonNull ImageProxy image) {
                    Bitmap bitmap = getBitmap(image);
                    processarImagem(bitmap);
                    image.close();
                }
                @Override
                public void onError(@NonNull ImageCaptureException exception) {
                    Log.e("CameraX", "Erro ao capturar foto: " + exception.getMessage());
                }
            });
        }
    }

    private Bitmap getBitmap(ImageProxy image) {
        byte[] bytes = convertImageToByteArray(image);
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    }
    private byte[] convertImageToByteArray(ImageProxy image) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        return bytes;
    }

    private void processarImagem(Bitmap bitmap) {
        MPImage mpImg = new BitmapImageBuilder(bitmap).build();
        HandLandmarkerResult resultado = handLandmark.detect(mpImg);
        float[] lista = new float[42];
        int x = 0;
        if (!resultado.landmarks().isEmpty()) {
            for (List<NormalizedLandmark> l : resultado.landmarks()) {
                for (NormalizedLandmark landmark : l) {
                    lista[x++] = landmark.x();
                    lista[x++] = landmark.y();
                }
            }
        }

        try {
            OrtSession session = ortEnvironment.createSession(modelBuffer, sessionOptions);
            OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(lista), new long[]{1, 42});
            OrtSession.Result result = session.run(Collections.singletonMap("input", tensor));
            String[] p = (String[]) result.get(0).getValue();
            runOnUiThread(() ->
                    textView.setText(p[0])
            );

            tensor.close();
            result.close();
            session.close();
        } catch (Exception e) {
            Log.e("ONNX", "Erro ao processar o modelo", e);
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

    private void startCamera(int cameraFacing) {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                imageCapture = new ImageCapture.Builder().build();
                CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(cameraFacing).build();
                cameraProvider.unbindAll();
                Log.d("info camera",""+cameraProvider.getCameraInfo(cameraSelector));
                cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageCapture);
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
            } catch (ExecutionException | InterruptedException e) {
                Log.e("CameraX", "Erro ao iniciar a câmera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }
}