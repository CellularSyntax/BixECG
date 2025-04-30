package com.example.mockecgviewer;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class TorchModelHandler {
    private final Module module;
    private final Context context;

    public TorchModelHandler(Context context, String modelPath) {
        this.context = context;
        try {
            module = Module.load(assetFilePath(modelPath));
        } catch (Exception e) {
            throw new RuntimeException("Error loading model", e);
        }
    }

    private String assetFilePath(String assetName) throws Exception {
        File file = new File(context.getCacheDir(), assetName);  // ✅ Now correct
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }

        return file.getAbsolutePath();
    }

    public float[] predict(float[] inputSequence, int seqLen) {
        Tensor inputTensor = Tensor.fromBlob(inputSequence, new long[]{1, seqLen, 1});
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        return outputTensor.getDataAsFloatArray();
    }
}
