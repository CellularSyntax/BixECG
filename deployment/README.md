# Deployment (Arduino Portenta H7 / Cortex-M7)

`firmware/` holds the firmware-format export of the **final, default weights**
(`bixecg/weights/bixecg_mlstm_2522_detaug.pt`, detector-anchored training + augmentation).

| File | Content |
|:-----|:--------|
| `firmware/bixecg_weights_aug.h` | C header: model `#define`s + one `static const float` array per tensor, plus `bixecg_all_weights[2522]` |
| `firmware/bixecg_weights_aug.bin` | Flat little-endian float32, tensors concatenated in PyTorch `state_dict` order (10,088 bytes) |
| `firmware/bixecg_weights_aug.json` | Manifest: dims, ordered tensor list (name, shape, offset, numel, per-tensor SHA-256), SHA-256 of the `.bin` |
| `firmware/export_fw_aug.py` | Exporter + pure-numpy recurrent-form reference forward used to verify the export |

`sha256(bixecg_weights_aug.bin) = 51f1bad43263e57ed8c7bba4e3e3b27ebd87e440b7da5a52516e091a61036c70`

## Re-generate / verify

```bash
pip install -e .
python deployment/firmware/export_fw_aug.py --out-dir /tmp/fw --verify
# -> exported 2522 params, numpy-vs-torch max|dlogit| ~1e-5, argmax agreement 1.0000
```

Re-running the exporter on the bundled weights reproduces the three shipped files byte-for-byte.
To export the **pilot-firmware** weights (`bixecg_mlstm_2522.pt`, the weights flashed for the
on-device pilot reported in the paper) use `--weights pilot --stem bixecg_weights_pilot`.

## Layout notes

* LayerNorm / MultiHeadLayerNorm gains are stored raw as in the `state_dict`; xlstm uses residual
  weights, so the effective gain is `1 + weight` (no bias, eps = 1e-5).
* `conv1d` is depthwise causal: `y[t] = b[c] + sum_{j<K} w[c,0,j] * x[t-(K-1)+j]` (zero history).
* `q/k/v_proj` are block-diagonal (8 heads of 2x2): `out[h*2+o] = sum_d W[h,o,d] * in[h*2+d]`.
* mLSTM cell: input/forget gates are `Linear(48 -> 1)` over `concat(q, k, v)`; stabilised recurrent
  form with eps = 1e-6; keys scaled by `1/sqrt(16)`.
* Bidirectional: the same stack runs forward and on the time-reversed window;
  `concat(fwd, bwd) -> output_proj -> 4 logits`.
* Input: 300 samples at 250 Hz, band-pass 0.5-40 Hz, per-window z-score; one beat per window,
  anchored at the Pan-Tompkins R peak shifted back by the R-to-QRS-onset offset
  (see `scripts/retrain_anchor_experiment.py`, `detect_segmented` / `delta_of`).
