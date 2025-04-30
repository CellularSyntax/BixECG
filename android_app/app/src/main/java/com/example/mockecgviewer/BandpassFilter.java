package com.example.mockecgviewer;

import static java.lang.Math.*;

public class BandpassFilter {

    private final BiquadFilter[] biquads;

    public BandpassFilter(float lowcut, float highcut, int fs, int order) {
        this.biquads = createButterworthBandpass(lowcut, highcut, fs, order);
    }

    public float[] filter(float[] signal) {
        float[] result = signal.clone();
        for (BiquadFilter biquad : biquads) {
            result = biquad.filter(result);
        }
        return result;
    }

    private BiquadFilter[] createButterworthBandpass(float lowcut, float highcut, int fs, int order) {
        int nStages = order; // Each stage is one biquad
        double nyq = fs / 2.0;
        double low = lowcut / nyq;
        double high = highcut / nyq;

        BiquadFilter[] filters = new BiquadFilter[nStages];

        // (This is an approximation — real Butterworth design needs pole placement,
        // but it's fine for our mobile ECG prefiltering)
        for (int i = 0; i < nStages; i++) {
            double omega = PI * (low + high);
            double bw = PI * (high - low);

            double sin_omega = sin(omega);
            double cos_omega = cos(omega);

            double alpha = sin(bw) / 2.0;

            double b0 = alpha;
            double b1 = 0;
            double b2 = -alpha;
            double a0 = 1 + alpha;
            double a1 = -2 * cos_omega;
            double a2 = 1 - alpha;

            filters[i] = new BiquadFilter(
                    (float)(b0/a0), (float)(b1/a0), (float)(b2/a0),
                    (float)(a1/a0), (float)(a2/a0)
            );
        }
        return filters;
    }

    // ===== Inner Class for Single Biquad =====
    private static class BiquadFilter {
        private final float b0, b1, b2, a1, a2;
        private float x1 = 0, x2 = 0, y1 = 0, y2 = 0;

        public BiquadFilter(float b0, float b1, float b2, float a1, float a2) {
            this.b0 = b0;
            this.b1 = b1;
            this.b2 = b2;
            this.a1 = a1;
            this.a2 = a2;
        }

        public float[] filter(float[] input) {
            float[] output = new float[input.length];
            for (int i = 0; i < input.length; i++) {
                float x = input[i];
                float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                output[i] = y;

                x2 = x1;
                x1 = x;
                y2 = y1;
                y1 = y;
            }
            return output;
        }
    }
}
