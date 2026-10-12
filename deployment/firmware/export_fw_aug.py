"""Firmware weight export for the 2,522-param Bi-mLSTM + pure-numpy recurrent-form reference
inference (mirrors the per-timestep on-device loop) used to verify the export against the PyTorch
forward pass.

Exports (default stem ``bixecg_weights_aug``):
  <stem>.h     C header: model dimension #defines + one `static const float` array per tensor
  <stem>.bin   flat little-endian float32, tensors concatenated in state_dict order
  <stem>.json  manifest: ordered tensor list (name, shape, dtype, offset, numel, sha256), dims

The files shipped next to this script were produced from ``bixecg_mlstm_2522_detaug.pt``
(the final, default weights; source checkpoint name ``model_train_aug.pt``).

Usage::

    # re-export the bundled final weights and verify against PyTorch
    python deployment/firmware/export_fw_aug.py --out-dir /tmp/fw --verify
    # export the pilot-firmware weights instead
    python deployment/firmware/export_fw_aug.py --weights pilot --stem bixecg_weights_pilot --out-dir /tmp/fw
"""
import argparse, json, hashlib, os, numpy as np, torch

from bixecg.models.bixecg_net import BIXECG_2522_KWARGS, BixECGNet

EMB, NBLK, NH, K, SEQ, NCLS = 8, 3, 1, 11, 300, 4
INNER = 16; DH = INNER // NH; LN_EPS = 1e-5; CELL_EPS = 1e-6


def build_model():
    return BixECGNet(**BIXECG_2522_KWARGS).eval()

# ---------------- export ----------------
def c_ident(name): return name.replace(".", "_")
def export(sd, stem="bixecg_weights_aug", src_name="model_train_aug.pt", out_dir="."):
    os.makedirs(out_dir, exist_ok=True); stem_path = os.path.join(out_dir, stem)
    order = list(sd.keys()); tensors = {k: sd[k].detach().cpu().contiguous().float().numpy() for k in order}
    flat = np.concatenate([t.ravel() for t in tensors.values()]).astype("<f4")
    open(f"{stem_path}.bin", "wb").write(flat.tobytes())
    man = {"source_checkpoint": src_name, "format": "flat little-endian float32, tensors concatenated in PyTorch state_dict order (row-major / C order)",
           "dims": {"BIXECG_DIM": EMB, "BIXECG_INNER_DIM": INNER, "BIXECG_HEADS": NH, "BIXECG_HEAD_DIM": DH, "BIXECG_N_BLOCKS": NBLK,
                    "BIXECG_CONV_K": K, "BIXECG_SEQ_LEN": SEQ, "BIXECG_NUM_CLASSES": NCLS, "BIXECG_QKV_HEADS": INNER // 2, "BIXECG_QKV_BLOCK": 2},
           "n_params": int(flat.size), "sha256_bin": hashlib.sha256(flat.tobytes()).hexdigest(),
           "notes": ["LayerNorm / MultiHeadLayerNorm gains are stored RAW as in the state_dict; xlstm uses residual weights: effective gain = 1 + weight (bias=False, eps=1e-5).",
                     "conv1d is depthwise causal: y[t] = b[c] + sum_{j<K} w[c,0,j] * x[t-(K-1)+j] (zero history).",
                     "q/k/v_proj weights are block-diagonal (LinearHeadwiseExpand): shape (QKV_HEADS=8, 2, 2), out[h*2+o] = sum_d W[h,o,d] * in[h*2+d].",
                     "mLSTM cell: igate/fgate Linear(48 -> NH) over concat(q,k,v); recurrent stabilised form with eps=1e-6; key scaled by 1/sqrt(HEAD_DIM).",
                     "Bidirectional: the SAME stack is run forward and on the time-reversed sequence; concat(fwd,bwd) -> output_proj -> 4 logits."],
           "tensors": []}
    off = 0; lines = []
    lines.append("// bixecg_weights_aug.h -- AUTO-GENERATED from %s (ours-aug 2,522-param Bi-mLSTM)\n" % src_name)
    lines.append("// Layout inferred from PORTENTA_ECG3_NEW.md (constants) -- original exporter unavailable. Values are raw state_dict float32.\n")
    lines.append("#pragma once\n#include <stdint.h>\n\n")
    for kname, v in man["dims"].items(): lines.append(f"#define {kname} {v}\n")
    lines.append(f"#define BIXECG_N_PARAMS {flat.size}\n\n")
    for k in order:
        t = tensors[k]; n = t.size
        man["tensors"].append({"name": k, "c_name": c_ident(k), "shape": list(t.shape), "dtype": "float32", "offset_elems": off, "offset_bytes": off * 4, "numel": int(n),
                               "sha256": hashlib.sha256(t.astype("<f4").tobytes()).hexdigest()})
        lines.append(f"// {k} shape={tuple(t.shape)} offset={off}\n")
        vals = ", ".join(f"{x:.9g}f" for x in t.ravel())
        lines.append(f"static const float {c_ident(k)}[{n}] = {{ {vals} }};\n")
        off += n
    lines.append("\n// All tensors concatenated (same order/offsets as bixecg_weights_aug.bin)\n")
    lines.append(f"static const float bixecg_all_weights[BIXECG_N_PARAMS] = {{ {', '.join(f'{x:.9g}f' for x in flat)} }};\n")
    open(f"{stem_path}.h", "w").write("".join(lines))
    json.dump(man, open(f"{stem_path}.json", "w"), indent=1)
    return man, flat

def load_exported(manifest_path, bin_path):
    man = json.load(open(manifest_path)); flat = np.frombuffer(open(bin_path, "rb").read(), dtype="<f4")
    assert flat.size == man["n_params"]
    return man, {t["name"]: flat[t["offset_elems"]:t["offset_elems"] + t["numel"]].reshape(t["shape"]).astype(np.float64) for t in man["tensors"]}

# ---------------- numpy reference inference (recurrent form, float64) ----------------
def silu(x): return x / (1.0 + np.exp(-x))
def logsigmoid(x): return -np.logaddexp(0.0, -x)
def layernorm(x, w_raw, eps=LN_EPS):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * (1.0 + w_raw)
def headwise(x, W):  # x (INNER,), W (H8,2,2): out[h*2+o]=sum_d W[h,o,d] x[h*2+d]
    H = W.shape[0]; xr = x.reshape(H, 2); return np.einsum("hd,hod->ho", xr, W).reshape(-1)
def run_stack(E, W):
    """E: (T,EMB) embedded sequence (already in the processing direction). Returns (T,EMB) after post_blocks_norm."""
    T = E.shape[0]; x = E.copy()
    for b in range(NBLK):
        p = f"xlstm_stack.blocks.{b}."
        ln_w = W[p + "xlstm_norm.weight"]; Wup = W[p + "xlstm.proj_up.weight"]; Wq = W[p + "xlstm.q_proj.weight"]; Wk = W[p + "xlstm.k_proj.weight"]; Wv = W[p + "xlstm.v_proj.weight"]
        cw = W[p + "xlstm.conv1d.conv.weight"][:, 0, :]; cb = W[p + "xlstm.conv1d.conv.bias"]
        Wi = W[p + "xlstm.mlstm_cell.igate.weight"]; bi = W[p + "xlstm.mlstm_cell.igate.bias"]; Wf = W[p + "xlstm.mlstm_cell.fgate.weight"]; bf = W[p + "xlstm.mlstm_cell.fgate.bias"]
        on_w = W[p + "xlstm.mlstm_cell.outnorm.weight"]; skip = W[p + "xlstm.learnable_skip"]; Wdown = W[p + "xlstm.proj_down.weight"]
        C = np.zeros((NH, DH, DH)); n = np.zeros((NH, DH)); m = np.zeros(NH); hist = np.zeros((K, INNER)); out = np.zeros_like(x)
        for t in range(T):
            xn = layernorm(x[t], ln_w)
            xin = Wup @ xn; xm, z = xin[:INNER], xin[INNER:]
            hist = np.roll(hist, -1, axis=0); hist[-1] = xm                     # causal ring buffer (oldest first)
            xc = (cw * hist.T).sum(1) + cb                                      # depthwise causal conv, w[c,j] * x[t-(K-1)+j]
            xa = silu(xc)
            q = headwise(xa, Wq); k = headwise(xa, Wk); v = headwise(xm, Wv)
            g_in = np.concatenate([q, k, v]); ig = Wi @ g_in + bi; fg = Wf @ g_in + bf      # (NH,)
            qh = q.reshape(NH, DH); kh = k.reshape(NH, DH) / np.sqrt(DH); vh = v.reshape(NH, DH)
            lf = logsigmoid(fg); m_new = np.maximum(lf + m, ig); fa = np.exp(lf + m - m_new); ia = np.exp(ig - m_new)
            C = fa[:, None, None] * C + ia[:, None, None] * (kh[:, :, None] * vh[:, None, :]); n = fa[:, None] * n + ia[:, None] * kh; m = m_new
            h_num = np.einsum("hd,hde->he", qh, C); qn = (qh * n).sum(-1)
            h = h_num / (np.maximum(np.abs(qn), np.exp(-m_new)) + CELL_EPS)[:, None]      # (NH,DH)
            # MultiHeadLayerNorm == group_norm over each head's DH values, weight (1+w) over NH*DH
            mu = h.mean(-1, keepdims=True); var = ((h - mu) ** 2).mean(-1, keepdims=True)
            hn = ((h - mu) / np.sqrt(var + LN_EPS)).reshape(-1) * (1.0 + on_w)
            hs = (hn + skip * xa) * silu(z)
            out[t] = Wdown @ hs
        x = x + out
    return layernorm(x, W["xlstm_stack.post_blocks_norm.weight"])
def numpy_forward(x300, W):
    """x300: (300,) z-normalised window -> (300,4) logits. Mirrors BixECGNet.forward."""
    E = x300[:, None] * W["input_proj.weight"][None, :, 0] + W["input_proj.bias"][None, :]
    f = run_stack(E, W); bwd = run_stack(E[::-1].copy(), W)[::-1]
    return np.concatenate([f, bwd], -1) @ W["output_proj.weight"].T + W["output_proj.bias"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="path to a .pt state_dict (overrides --weights)")
    ap.add_argument("--weights", default="detaug", help="bundled weight set: detaug (default) or pilot")
    ap.add_argument("--stem", default="bixecg_weights_aug", help="output file stem")
    ap.add_argument("--src-name", default=None, help="source checkpoint name recorded in the header/manifest")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--verify", action="store_true", help="check numpy recurrent reference vs PyTorch forward")
    a = ap.parse_args()
    if a.checkpoint:
        sd = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        sd = sd.get("model_state_dict", sd)
        src = a.src_name or os.path.basename(a.checkpoint)
    else:
        from bixecg.pretrained import load_state_dict
        sd = load_state_dict(a.weights)
        src = a.src_name or ("model_train_aug.pt" if a.weights in ("detaug", "default", "final") else "bixecg_mlstm_2522.pt")
    man, flat = export(sd, stem=a.stem, src_name=src, out_dir=a.out_dir)
    print(f"exported {man['n_params']} params  sha256(bin)={man['sha256_bin']}")
    if a.verify:
        model = build_model(); model.load_state_dict(sd, strict=True)
        _, W = load_exported(os.path.join(a.out_dir, a.stem + ".json"), os.path.join(a.out_dir, a.stem + ".bin"))
        rng = np.random.default_rng(0); x = rng.standard_normal(SEQ)
        with torch.no_grad():
            ref = model(torch.tensor(x, dtype=torch.float32)[None, :, None])[0].numpy()
        got = numpy_forward(x, W)
        err = float(np.abs(got - ref).max()); agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
        print(f"numpy-vs-torch max|dlogit|={err:.2e}  argmax agreement={agree:.4f}")
        if err > 1e-3:
            raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()
