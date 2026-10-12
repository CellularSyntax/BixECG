"""Detector-anchored training + evaluation protocol of the final BixECG model (v1.1.0).

Trains the selected 2,522-parameter Bi-mLSTM (:class:`bixecg.models.bixecg_net.BixECGNet`) on QTDB
windows built with one of three anchorings, then evaluates on external streams under both
annotation-anchored and detector-anchored windowing:

  ann  annotation-anchored windows [QRS onset_i, QRS onset_i+1) (earlier recipe; pilot-firmware weights)
  det  detector-anchored windows: segmented adaptive Pan-Tompkins R peaks shifted back by the median
       R-to-QRS-onset offset (the deployment windowing)
  aug  det + augmentation (FINAL recipe -> bixecg_mlstm_2522_detaug.pt): 3 extra anchor-jittered copies
       per beat (sigma 40 ms, clip +-60 ms) and on-the-fly +-10 % time-stretch, 0.1-0.5 Hz baseline
       wander (<=0.3 z) and Gaussian noise (sd <=0.15 z)

Metric: tolerance-aware macro-F1 over {NW, P, QRS, T} (tolerances P 52 ms, QRS 48 ms, T 124 ms),
QRS F1 and Cohen's kappa.

Inputs (no data are distributed with this repository):

  --qtdb-csv   headerless CSV with columns ecg,label,patient,lead (250 Hz, labels 0=NW 1=P 2=QRS 3=T);
               this is the ``all_ecg_data.csv`` file used by ``train_model.py``.
  --eval       NAME=PATH.npz, repeatable. Each .npz holds ``x`` (1-D raw single-lead ECG at 250 Hz)
               and ``y`` (per-sample labels 0-3, same length) for one evaluation stream.
  --det-band   NAME=LO,HI detector band-pass (Hz) per eval stream; default 5,15. The paper used
               5,15 for LUDB and 8,18 for both LVAD-DB raters.
  --pool       GROUP=NAME1,NAME2 additionally report a pooled score (e.g. LVAD=lvad_rater1,lvad_rater2).

Example::

    python scripts/retrain_anchor_experiment.py --anchor aug --qtdb-csv data/QTDataset/all_ecg_data.csv \
        --eval ludb=data/ludb_stream.npz --out runs/aug
"""
import argparse, os, json, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import f1_score, cohen_kappa_score
from bixecg.utils.validation_helper_fns import apply_bandpass_filter, get_tolerance_map
from bixecg.utils.helper_fns import z_normalize, apply_classwise_tolerance_matching, compute_class_weights
from bixecg.utils.scheduler import WarmupCosineScheduler
from bixecg.models.bixecg_net import BixECGNet, BIXECG_2522_KWARGS

# ---- canonical constants (manuscript / firmware signal chain) ----
FS=250; SEQ_LEN=300; BANDPASS=(0.5,40.0); QRS_LABEL=2

# ---- run configuration (overridden by CLI in main()) ----
QTDB_CSV=None; EPOCHS=40
dev="cuda" if torch.cuda.is_available() else "cpu"
TOL=get_tolerance_map(); IGN=-100
DET_BAND_QTDB=(5,15); DEFAULT_EVAL_BAND=(5,15)
seed=42
# augmentation arm params
AUG_K=3; JIT_SD=int(0.040*FS); JIT_MAX=int(0.060*FS)      # 3 jittered copies/beat, sigma 40 ms, clip 60 ms
NOISE_SD_MAX=0.15; WANDER_AMP=0.3; STRETCH=0.10           # z-units; z-units; +-10 %
AUGMENT=False

def augment_batch(xb,yb):
    """VECTORIZED: xb (B,T,1) z-normed, yb (B,T) long. Time-stretch (+-STRETCH, nearest labels) +
    baseline wander + gaussian noise, all restricted to the valid (non-pad) region."""
    B,T,_=xb.shape; dev_=xb.device
    valid=(yb!=IGN).sum(1).clamp(min=2)                                   # (B,)
    f=torch.empty(B,device=dev_).uniform_(1-STRETCH,1+STRETCH)
    L=(valid.float()*f).round().long().clamp(2,T)                          # stretched lengths
    t=torch.arange(T,device=dev_,dtype=torch.float32)[None,:]              # (1,T)
    src=t*((valid-1).float()/(L-1).float().clamp(min=1))[:,None]           # source index per output pos
    src=src.clamp(max=(valid-1).float()[:,None])
    inmask=t<L.float()[:,None]
    i0=src.floor().long().clamp(0,T-1); i1=(i0+1).clamp(max=T-1); w=src-i0.float()
    x=xb[:,:,0]
    xs=torch.gather(x,1,i0)*(1-w)+torch.gather(x,1,i1)*w
    ys=torch.gather(yb,1,src.round().long().clamp(0,T-1))
    xs=torch.where(inmask,xs,torch.zeros_like(xs)); ys=torch.where(inmask,ys,torch.full_like(ys,IGN))
    # baseline wander (0.1-0.5 Hz) + gaussian noise on the valid region only
    tt=t/FS
    fr=torch.empty(B,1,device=dev_).uniform_(0.1,0.5); ph=torch.empty(B,1,device=dev_).uniform_(0,6.283)
    amp=torch.empty(B,1,device=dev_).uniform_(0,WANDER_AMP)
    wander=amp*torch.sin(2*3.14159*fr*tt+ph)
    sd=torch.empty(B,1,device=dev_).uniform_(0,NOISE_SD_MAX)
    noise=torch.randn(B,T,device=dev_)*sd
    xs=xs+(wander+noise)*inmask.float()
    return xs[:,:,None],ys

def seed_all():
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

# ---- windowing (verbatim from comprehensive_rescore) ----
def extract_from_onsets(xb,y,onsets,beats=1,max_len=SEQ_LEN,pad=IGN):
    onsets=np.asarray(sorted(int(o) for o in onsets if o>=0)); X=[];Y=[];S=[]
    for i in range(len(onsets)-beats):
        s=onsets[i];e=onsets[i+beats]
        if e<=s or e-s>max_len or e-s<10: continue
        xs=xb[s:e].astype(float);ys=y[s:e].astype(int)
        if len(xs)<max_len:
            xs=np.pad(xs,(0,max_len-len(xs)),constant_values=0.0);ys=np.pad(ys,(0,max_len-len(ys)),constant_values=pad)
        X.append(xs);Y.append(ys);S.append(s)
    return np.asarray(X),np.asarray(Y),np.asarray(S)
def _mwi(sig,fs,lo,hi):
    ny=fs/2.;b,a=butter(2,[lo/ny,hi/ny],btype="band");f=filtfilt(b,a,sig)
    d=np.ediff1d(f,to_begin=0);sq=d*d;w=int(0.15*fs);return np.convolve(sq,np.ones(w)/w,mode="same")
def _pt_seg(sig,fs,band,thr=0.25,refr=0.2):
    m=_mwi(sig,fs,band[0],band[1]);cand,_=find_peaks(m,distance=int(refr*fs))
    if len(cand)<1:return []
    init=m[:int(2*fs)];SPKI=0.25*(init.max() if init.size else m.max());NPKI=0.125*(init.mean() if init.size else m.mean());acc=[]
    for p in cand:
        T=NPKI+thr*(SPKI-NPKI)
        if m[p]>=T:acc.append(p);SPKI=0.125*m[p]+0.875*SPKI
        else:NPKI=0.125*m[p]+0.875*NPKI
    if len(acc)>2:
        rr=np.diff(acc);md=np.median(rr);T2=0.5*(NPKI+thr*(SPKI-NPKI));ex=[]
        for i in range(len(acc)-1):
            a0,a1=acc[i],acc[i+1]
            if a1-a0>1.66*md:
                sg=[q for q in cand if a0<q<a1 and m[q]>T2]
                if sg:ex.append(max(sg,key=lambda q:m[q]))
        acc=sorted(set(acc+list(ex)))
    return acc
def detect_segmented(xb,onsets,fs,band,gap_s=1.5,pad_s=0.4):
    if len(onsets)==0: return np.array([])
    gap=int(gap_s*fs);pad=int(pad_s*fs);bnds=np.where(np.diff(onsets)>gap)[0]
    starts=[onsets[0]]+[onsets[i+1] for i in bnds];ends=[onsets[i] for i in bnds]+[onsets[-1]];r=[]
    for s,e in zip(starts,ends):
        s0=max(0,s-pad);e0=min(len(xb),e+pad);seg=xb[s0:e0]
        if len(seg)<int(0.5*fs):continue
        for p in _pt_seg(seg,fs,band):
            q=s0+p;a=max(0,q-int(0.05*fs));b_=min(len(xb),q+int(0.05*fs))
            if b_>a:r.append(a+int(np.argmax(np.abs(xb[a:b_]))))
    return np.array(sorted(set(r)))
def onsets_of(y): return np.where(np.diff(np.concatenate([[0],(y==QRS_LABEL).astype(int)]))==1)[0]
def delta_of(xb,onsets):
    qd=int(0.08*FS)
    ds=[int(np.argmax(np.abs(np.gradient(xb[o:min(o+qd,len(xb))])))) for o in onsets if min(o+qd,len(xb))-o>=3]
    return int(np.median(ds)) if ds else int(0.02*FS)

# ---- build QTDB windows per (patient,lead) block ----
def build_qtdb(anchor):
    df=pd.read_csv(QTDB_CSV,header=None,names=["ecg","label","patient","lead"])
    ecg=df["ecg"].to_numpy(float); lab=df["label"].to_numpy(int); pid=df["patient"].to_numpy(int); lead=df["lead"].to_numpy(int)
    key=pid.astype(np.int64)*100+lead.astype(np.int64)
    chg=np.where(np.diff(key)!=0)[0]+1; bstarts=np.concatenate([[0],chg]); bends=np.concatenate([chg,[len(key)]])
    X=[];Y=[];M=[]
    for s,e in zip(bstarts,bends):
        x=ecg[s:e]; y=lab[s:e]
        if len(x)<int(1.5*FS): continue
        xb=apply_bandpass_filter(x,lowcut=BANDPASS[0],highcut=BANDPASS[1],fs=FS)
        ons=onsets_of(y)
        if len(ons)<2: continue
        if anchor=="ann":
            Xi,Yi,_=extract_from_onsets(xb,y,ons)
        elif anchor=="aug":
            # detector-anchored + ANCHOR-JITTER: clean det windows + K jittered copies (sigma 40 ms, clip +-60 ms)
            dl=delta_of(xb,ons); rdet=detect_segmented(xb,ons,FS,DET_BAND_QTDB)
            if len(rdet)<2: continue
            base=rdet-dl; rng=np.random.default_rng(seed+int(pid[s]))
            parts=[]
            for k in range(AUG_K+1):
                jit=np.zeros(len(base),int) if k==0 else np.clip(rng.normal(0,JIT_SD,len(base)),-JIT_MAX,JIT_MAX).astype(int)
                Xk,Yk,_=extract_from_onsets(xb,y,base+jit)
                if Yk.ndim==2 and len(Xk): parts.append((Xk,Yk))
            if not parts: continue
            Xi=np.concatenate([p[0] for p in parts]); Yi=np.concatenate([p[1] for p in parts])
            keep=(Yi==QRS_LABEL).any(axis=1); Xi,Yi=Xi[keep],Yi[keep]
        else:
            dl=delta_of(xb,ons); rdet=detect_segmented(xb,ons,FS,DET_BAND_QTDB)
            if len(rdet)<2: continue
            Xi,Yi,_=extract_from_onsets(xb,y,rdet-dl)
            if Yi.ndim!=2 or len(Yi)==0: continue
            keep=(Yi==QRS_LABEL).any(axis=1)   # keep only windows over annotated beats
            Xi,Yi=Xi[keep],Yi[keep]
        if len(Xi)==0: continue
        Xi=np.array([z_normalize(w) for w in Xi])
        X.append(Xi);Y.append(Yi);M.append(np.full(len(Xi),pid[s]))
    X=np.concatenate(X);Y=np.concatenate(Y);M=np.concatenate(M)
    return X.astype(np.float32),Y.astype(np.int64),M

def split_by_patient(X,Y,M,frac=0.8):
    pats=np.unique(M); rng=np.random.default_rng(seed); rng.shuffle(pats)
    ntr=int(len(pats)*frac); tr=set(pats[:ntr].tolist())
    mtr=np.array([p in tr for p in M])
    return X[mtr],Y[mtr],X[~mtr],Y[~mtr]

# ---- model / train ----
def build(): return BixECGNet(**BIXECG_2522_KWARGS).to(dev)
def fwd(model,xb):
    o=model(xb); return o if o.shape[-1]==4 else o.transpose(1,2)   # (B,T,4)
def predict(model,X):
    model.eval();P=[]
    with torch.no_grad():
        for i in range(0,len(X),512):
            xb=torch.tensor(X[i:i+512][:,:,None],dtype=torch.float32,device=dev)
            P.append(fwd(model,xb).argmax(-1).cpu().numpy())
    return np.concatenate(P) if P else np.zeros((0,SEQ_LEN),int)

def tmatch(yt,yp):
    try: return apply_classwise_tolerance_matching(yt,yp,FS,TOL)
    except TypeError: return apply_classwise_tolerance_matching(yt,yp,TOL,FS)
def macro_and_qrs(yt,yp):
    ypt=tmatch(yt,yp); m=yt!=IGN; a=yt[m];b=ypt[m]
    mf1=f1_score(a,b,labels=[0,1,2,3],average="macro",zero_division=0)
    qrs=f1_score((a==2).astype(int),(b==2).astype(int),zero_division=0)
    return float(mf1),float(qrs),float(cohen_kappa_score(a,b))

def train(Xtr,Ytr,Xval,Yval,ckpt_path,patience=8):
    seed_all(); model=build()
    cw=compute_class_weights(Ytr,4); cw[0]*=2.0
    lossf=nn.CrossEntropyLoss(ignore_index=IGN,weight=torch.tensor(cw,dtype=torch.float32).to(dev),label_smoothing=0.05)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sch=WarmupCosineScheduler(opt,warmup_epochs=2,total_epochs=EPOCHS,min_lr=1e-5)
    dl=DataLoader(TensorDataset(torch.tensor(Xtr[:,:,None]),torch.tensor(Ytr)),batch_size=256,shuffle=True)
    best=-1;best_sd=None;since=0
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in dl:
            xb=xb.float().to(dev);yb=yb.long().to(dev)
            if AUGMENT: xb,yb=augment_batch(xb,yb)
            opt.zero_grad();logit=fwd(model,xb)
            loss=lossf(logit.reshape(-1,4),yb.reshape(-1));loss.backward();opt.step()
        sch.step()
        vp=predict(model,Xval); vf1,_,_=macro_and_qrs(Yval,vp)
        if vf1>best:
            best=vf1;best_sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_sd,ckpt_path);since=0   # persist best immediately (walltime-safe)
        else: since+=1
        print(f"    ep{ep} val_tolMacroF1={vf1:.4f} best={best:.4f}",flush=True)
        if since>=patience: print(f"    early stop at ep{ep}",flush=True);break
    model.load_state_dict(best_sd);return model,best

# ---- eval external ----
def load_stream(path):
    d=np.load(path); x=np.asarray(d["x"],float).ravel(); y=np.asarray(d["y"],int).ravel()
    assert len(x)==len(y), f"{path}: x/y length mismatch {len(x)} vs {len(y)}"
    return x,y
def build_eval(path,band,anchor):
    x,y=load_stream(path);xb=apply_bandpass_filter(x,lowcut=BANDPASS[0],highcut=BANDPASS[1],fs=FS)
    ons=onsets_of(y)
    if anchor=="ann": X,Y,_=extract_from_onsets(xb,y,ons)
    else:
        dl=delta_of(xb,ons);rdet=detect_segmented(xb,ons,FS,band);X,Y,_=extract_from_onsets(xb,y,rdet-dl)
    X=np.array([z_normalize(w) for w in X]).astype(np.float32);return X,Y.astype(np.int64)
def eval_model(model,anchor,evals,bands,pools):
    res={};cache={}
    for name,path in evals.items():
        X,Y=build_eval(path,bands.get(name,DEFAULT_EVAL_BAND),anchor);cache[name]=(X,Y)
        res[name]=macro_and_qrs(Y,predict(model,X))
    for grp,names in pools.items():
        Xr=np.concatenate([cache[n][0] for n in names]);Yr=np.concatenate([cache[n][1] for n in names])
        res[grp]=macro_and_qrs(Yr,predict(model,Xr))
    return res

def _kv(items,what):
    out={}
    for it in items or []:
        if "=" not in it: raise SystemExit(f"--{what} expects NAME=VALUE, got {it!r}")
        k,v=it.split("=",1);out[k.strip()]=v.strip()
    return out

def main(argv=None):
    global QTDB_CSV,EPOCHS,AUGMENT,seed
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qtdb-csv",required=True,help="QTDB all_ecg_data.csv (ecg,label,patient,lead; no header)")
    ap.add_argument("--out",required=True,help="output directory for checkpoints + results JSON")
    ap.add_argument("--anchor",action="append",choices=["ann","det","aug"],
                    help="training anchoring (repeatable); default: ann and det. Final model: aug")
    ap.add_argument("--eval",action="append",default=[],metavar="NAME=PATH.npz",help="evaluation stream (repeatable)")
    ap.add_argument("--det-band",action="append",default=[],metavar="NAME=LO,HI",help="detector band per eval stream")
    ap.add_argument("--pool",action="append",default=[],metavar="GROUP=N1,N2",help="pooled evaluation group")
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(argv)
    if a.epochs<=2: raise SystemExit("--epochs must exceed the 2 warm-up epochs")
    QTDB_CSV=a.qtdb_csv;EPOCHS=a.epochs;seed=a.seed;OUT=a.out;os.makedirs(OUT,exist_ok=True)
    evals=_kv(a.eval,"eval")
    bands={k:tuple(float(t) for t in v.split(",")) for k,v in _kv(a.det_band,"det-band").items()}
    pools={k:[n.strip() for n in v.split(",")] for k,v in _kv(a.pool,"pool").items()}
    for g,ns in pools.items():
        miss=[n for n in ns if n not in evals]
        if miss: raise SystemExit(f"--pool {g}: unknown eval names {miss}")
    RES={}
    ANCHORS=a.anchor or ["ann","det"]
    for train_anchor in ANCHORS:
        AUGMENT=(train_anchor=="aug")
        print(f"=== TRAIN anchor={train_anchor} augment={AUGMENT} ===",flush=True)
        X,Y,M=build_qtdb(train_anchor)
        print(f"  qtdb windows={len(X)} patients={len(np.unique(M))}",flush=True)
        Xtr,Ytr,Xval,Yval=split_by_patient(X,Y,M)
        model,bv=train(Xtr,Ytr,Xval,Yval,os.path.join(OUT,f"model_train_{train_anchor}.pt"))
        RES[train_anchor]={"val_bestF1":bv}
        for eval_anchor in ["ann","det"]:
            if not evals: break
            r=eval_model(model,eval_anchor,evals,bands,pools);RES[train_anchor][f"eval_{eval_anchor}"]=r
            msg="  ".join(f"{k} mF1/QRS/kappa={v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}" for k,v in r.items())
            print(f"  [train {train_anchor} | eval {eval_anchor}] {msg}",flush=True)
        json.dump(RES,open(os.path.join(OUT,"retrain_anchor_results.json"),"w"),indent=1)   # partial dump (walltime-safe)
    print("RETRAIN_EXPERIMENT_DONE",flush=True)


if __name__=="__main__":
    main()
