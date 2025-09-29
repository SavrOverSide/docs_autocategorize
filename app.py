import json
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# чтение pdf — опционально
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# где разрешаем читать/писать (пробрасывается в docker через -e DOCS_BASE=/data)
ALLOWED_BASE = Path(os.environ.get("DOCS_BASE", "/data")).resolve()

app = FastAPI(title="Doc Categorizer API")

# ----------------------------- FS helpers -----------------------------

def _normalize_user_path(s: str) -> Path:
    """Относительные пути считаем относительно ALLOWED_BASE; ~ разворачиваем."""
    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = ALLOWED_BASE / p
    return p

def _resolve_inside_base(p: Path) -> Path:
    rp = p.resolve()
    if not str(rp).startswith(str(ALLOWED_BASE)):
        raise HTTPException(status_code=400, detail=f"path outside allowed base: {rp}")
    return rp

def _write_uploads_to_dir(files: Optional[List[UploadFile]], dst_dir: Path):
    if not files:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.filename:
            continue
        target = dst_dir / Path(f.filename).name
        with target.open("wb") as out:
            shutil.copyfileobj(f.file, out)

def _extract_zip_to_dir(bundle: UploadFile, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = dst_dir / "__bundle__.zip"
    with tmp_zip.open("wb") as out:
        shutil.copyfileobj(bundle.file, out)
    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall(dst_dir)
    tmp_zip.unlink(missing_ok=True)

def _run_cmd(cmd: list, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, env=env
    )
    return proc.returncode, proc.stdout, proc.stderr

def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

# ------------------------------ file reading ------------------------------

READ_TXT_EXT = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".rtf"
}
READ_PDF_EXT = {".pdf"}

def _read_text_from_file(p: Path, max_chars: int = 15000) -> str:
    suf = p.suffix.lower()
    try:
        if suf in READ_TXT_EXT:
            t = p.read_text(encoding="utf-8", errors="ignore")
            return t[:max_chars]
        elif suf in READ_PDF_EXT and pdf_extract_text is not None:
            t = pdf_extract_text(str(p)) or ""
            return t[:max_chars]
        else:
            t = p.read_text(encoding="utf-8", errors="ignore")
            return t[:max_chars]
    except Exception:
        return ""

def _iter_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.is_file()]

# ------------------------------ models ------------------------------

class FitResponse(BaseModel):
    run_dir: str
    categories: dict
    assignments: List[dict]
    predictions: List[dict] = []
    doc_types: List[str] = []
    logs: str

class PredictResponse(BaseModel):
    predictions: List[dict]
    logs: str

class ClassifyItem(BaseModel):
    file: str
    scores: Dict[str, float]
    pred_label: str
    pred_prob: float
    routed_to: Optional[str] = None

class ClassifyResponse(BaseModel):
    items: List[ClassifyItem]
    logs: str

# ---------------------- simple similarity/NLI classifiers ----------------

def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x / max(1e-8, temp)) - np.max(x)
    e = np.exp(z); s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0/len(e))

class ZeroShotNLI:
    def __init__(self, model_name: str, device: str = "cpu",
                 hypothesis: str = "Этот текст относится к категории {}."):
        dev_idx = -1
        if device.startswith("cuda"):
            try:
                dev_idx = int(device.split(":")[1])
            except Exception:
                dev_idx = 0
        self.pipe = pipeline("zero-shot-classification",
                             model=model_name, device=dev_idx, truncation=True)
        self.hypothesis = hypothesis

    def classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        if not text.strip():
            return {lbl: 0.0 for lbl in labels}
        res = self.pipe(text, candidate_labels=labels,
                        hypothesis_template=self.hypothesis, multi_label=False)
        order = res["labels"]; scores = res["scores"]
        return {lbl: float(scores[order.index(lbl)]) if lbl in order else 0.0
                for lbl in labels}

class SimilarityClassifier:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 device: str = "cpu",
                 label_prompt: str = "Класс документа: {}",
                 temperature: float = 0.07):
        self.model = SentenceTransformer(model_name, device=device)
        self.label_prompt = label_prompt
        self.temperature = temperature

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)

    def classify(self, text: str, labels: List[str]) -> Dict[str, float]:
        if not text.strip():
            return {lbl: 0.0 for lbl in labels}
        doc = self._embed([text])[0]
        labs = self._embed([self.label_prompt.format(lbl) for lbl in labels])
        sims = labs @ doc
        probs = _softmax(sims, temp=self.temperature)
        return {lbl: float(p) for lbl, p in zip(labels, probs)}

# ----------------------------- endpoints ----------------------------

@app.get("/health")
def health():
    return {"ok": True, "base": str(ALLOWED_BASE)}

@app.post("/fit", response_model=FitResponse)
async def fit(
    # входные данные
    files: Optional[List[UploadFile]] = File(default=None),
    bundle: Optional[UploadFile] = File(default=None),
    input_dir: Optional[str] = Form(default=None),
    paths: Optional[List[str]] = Form(default=None),

    # куда сохранять артефакты
    out_root: str = Form("runs"),  # будет /data/runs/<run_id>

    # параметры пайплайна auto_categorize.py
    device: str = Form("cpu"),
    min_k: int = Form(7),
    max_k: int = Form(12),
    k_select_tol: float = Form(0.25),
    recluster_large: bool = Form(True),
    recluster_max_depth: int = Form(1),
    recluster_min_size: int = Form(6),
    recluster_child_min_size: int = Form(2),
    recluster_min_k: int = Form(2),
    recluster_max_k: int = Form(4),
    recluster_tol: float = Form(0.05),
    recluster_max_dominance: float = Form(0.90),
    df_stop_threshold: float = Form(0.6),
    label_local_stop_topk: int = Form(2),
    model_name: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    tags_backend: str = Form("yake"),
    label_use_yake: bool = Form(True),

    # one-shot predict + тип документа
    predict_on_fit: bool = Form(True),
    prob_temperature: float = Form(0.8),
    doc_types: str = Form(""),  # пусто = дефолтный список в скрипте
    doc_types_temperature: float = Form(0.7),

    # НОВОЕ: возможность зафиксировать k (например k=1 для одиночного файла)
    k: Optional[int] = Form(default=None),
):
    if not files and not bundle and not input_dir and not paths:
        raise HTTPException(status_code=400, detail="Provide files[], bundle, input_dir or paths[]")

    # собрать вход
    sid = uuid.uuid4().hex[:8]
    if input_dir:
        indir = _resolve_inside_base(_normalize_user_path(input_dir))
        if not indir.is_dir():
            raise HTTPException(status_code=400, detail=f"input_dir not found: {indir}")
    else:
        indir = Path(tempfile.gettempdir()) / f"docsvc_{sid}" / "input"
        indir.mkdir(parents=True, exist_ok=True)
        if paths:
            for s in paths:
                sp = _resolve_inside_base(_normalize_user_path(s))
                if not sp.is_file():
                    raise HTTPException(status_code=400, detail=f"path not found: {sp}")
                shutil.copyfile(sp, indir / sp.name)
        else:
            if files:
                _write_uploads_to_dir(files, indir)
            if bundle:
                _extract_zip_to_dir(bundle, indir)

    run_dir = _resolve_inside_base(_normalize_user_path(out_root)) / sid
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "python", "auto_categorize.py", "fit", str(indir),
            "--device", device,
            "--min-k", str(min_k), "--max-k", str(max_k), "--k-select-tol", str(k_select_tol),
            "--df-stop-threshold", str(df_stop_threshold),
            "--label-local-stop-topk", str(label_local_stop_topk),
            "--model-name", model_name,
            "--tags-backend", tags_backend,
            "--out-cats", str(run_dir / "categories.json"),
            "--out-assign", str(run_dir / "assignments.jsonl"),
            "--save-model", str(run_dir / "semantic_centroids.json"),
        ]

        # НОВОЕ: фиксированный k
        if k is not None:
            cmd += ["--k", str(k)]
            
        if recluster_large:
            cmd += [
                "--recluster-large",
                "--recluster-max-depth", str(recluster_max_depth),
                "--recluster-min-size", str(recluster_min_size),
                "--recluster-child-min-size", str(recluster_child_min_size),
                "--recluster-min-k", str(recluster_min_k),
                "--recluster-max-k", str(recluster_max_k),
                "--recluster-tol", str(recluster_tol),
                "--recluster-max-dominance", str(recluster_max_dominance),
            ]
        if label_use_yake:
            cmd += ["--label-use-yake"]

        if predict_on_fit:
            cmd += [
                "--predict-on-fit",
                "--predict-out", str(run_dir / "predictions.jsonl"),
                "--prob-temperature", str(prob_temperature),
                "--doc-types", doc_types,
                "--doc-types-temperature", str(doc_types_temperature),
                "--doc-types-out", str(run_dir / "doc_types.txt"),
            ]

        code, so, se = _run_cmd(cmd, cwd=Path("/app"))
        if code != 0:
            raise HTTPException(status_code=500, detail=f"Pipeline failed:\nSTDOUT:\n{so}\nSTDERR:\n{se}")

        cats = json.loads((run_dir / "categories.json").read_text(encoding="utf-8"))
        assigns = _read_jsonl(run_dir / "assignments.jsonl")
        preds = _read_jsonl(run_dir / "predictions.jsonl") if predict_on_fit else []
        dto = (run_dir / "doc_types.txt")
        doc_types_used = dto.read_text(encoding="utf-8").splitlines() if (predict_on_fit and dto.exists()) else []

        logs = (so or "") + (("\n" + se) if se else "")
        return JSONResponse(content={
            "run_dir": str(run_dir),
            "categories": cats,
            "assignments": assigns,
            "predictions": preds,
            "doc_types": doc_types_used,
            "logs": logs
        })
    finally:
        if not input_dir:
            try:
                tmp_root = indir.parents[1]
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

@app.post("/predict", response_model=PredictResponse)
async def predict(
    files: Optional[List[UploadFile]] = File(default=None),
    bundle: Optional[UploadFile] = File(default=None),
    input_dir: Optional[str] = Form(default=None),
    paths: Optional[List[str]] = Form(default=None),

    model: Optional[UploadFile] = File(default=None),
    model_path: Optional[str] = Form(default=None),

    device: str = Form("cpu"),
    sim_threshold: Optional[float] = Form(None),
    model_name: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    tags_backend: str = Form("none"),

    # zero-shot тип документа и температура для кластеров
    prob_temperature: float = Form(1.0),
    doc_types: str = Form(""),
    doc_types_temperature: float = Form(0.7),
):
    if not files and not bundle and not input_dir and not paths:
        raise HTTPException(status_code=400, detail="Provide files[], bundle, input_dir or paths[]")
    if not model and not model_path:
        raise HTTPException(status_code=400, detail="Provide model (file) or model_path")

    work = Path(tempfile.gettempdir()) / f"docsvc_pred_{uuid.uuid4().hex[:8]}"
    indir = work / "input"
    outdir = work / "out"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    local_model = work / "model.json"

    try:
        # вход
        if input_dir:
            p = _resolve_inside_base(_normalize_user_path(input_dir))
            if not p.is_dir():
                raise HTTPException(status_code=400, detail=f"input_dir not found: {p}")
            indir = p
        elif paths:
            for s in paths:
                sp = _resolve_inside_base(_normalize_user_path(s))
                if not sp.is_file():
                    raise HTTPException(status_code=400, detail=f"path not found: {sp}")
                shutil.copyfile(sp, indir / sp.name)
        else:
            if files:
                _write_uploads_to_dir(files, indir)
            if bundle:
                _extract_zip_to_dir(bundle, indir)

        # модель
        if model_path:
            mp = _resolve_inside_base(_normalize_user_path(model_path))
            if not mp.is_file():
                raise HTTPException(status_code=400, detail=f"model_path not found: {mp}")
            model_arg = str(mp)
        else:
            with local_model.open("wb") as out:
                shutil.copyfileobj(model.file, out)
            model_arg = str(local_model)

        cmd = [
            "python", "auto_categorize.py", "predict", str(indir),
            "--model", model_arg,
            "--device", device,
            "--model-name", model_name,
            "--tags-backend", tags_backend,
            "--prob-temperature", str(prob_temperature),
            "--doc-types", doc_types,
            "--doc-types-temperature", str(doc_types_temperature),
            "--out", str(outdir / "predictions.jsonl"),
        ]
        if sim_threshold is not None:
            cmd += ["--sim-threshold", str(sim_threshold)]

        code, so, se = _run_cmd(cmd, cwd=Path("/app"))
        if code != 0:
            raise HTTPException(status_code=500, detail=f"Predict failed:\nSTDOUT:\n{so}\nSTDERR:\n{se}")

        preds = _read_jsonl(outdir / "predictions.jsonl")
        logs = (so or "") + (("\n" + se) if se else "")
        return JSONResponse(content={"predictions": preds, "logs": logs})
    finally:
        shutil.rmtree(work, ignore_errors=True)

# -------------------- universal label classification -----------------

classify_help = """
/classify — прямое отнесение к предоставленным меткам (без обучения).
backend=sim — мультиязычные эмбеддинги (быстро), backend=nli — zero-shot NLI (тяжелее).
"""

@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    files: Optional[List[UploadFile]] = File(default=None),
    bundle: Optional[UploadFile] = File(default=None),
    input_dir: Optional[str] = Form(default=None),
    paths: Optional[List[str]] = Form(default=None),

    labels: Optional[List[str]] = Form(default=None),
    labels_json: Optional[str] = Form(default=None),

    backend: str = Form("sim"),  # sim | nli
    device: str = Form("cpu"),
    nli_model: str = Form("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
    sim_model: str = Form("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    hypothesis: str = Form("Этот текст относится к категории {}."),
    label_prompt: str = Form("Класс документа: {}"),
    temperature: float = Form(0.07),
):
    # собрать метки
    lbls: List[str] = []
    if labels:
        lbls = [str(x) for x in labels]
    elif labels_json:
        try:
            data = json.loads(labels_json)
            if isinstance(data, list):
                lbls = [str(x) for x in data]
        except Exception:
            pass
    if not lbls:
        raise HTTPException(status_code=400, detail="Provide labels[] or labels_json")

    sid = uuid.uuid4().hex[:8]
    indir = Path(tempfile.gettempdir()) / f"docsvc_cls_{sid}"
    indir.mkdir(parents=True, exist_ok=True)

    try:
        if input_dir:
            p = _resolve_inside_base(_normalize_user_path(input_dir))
            if not p.is_dir():
                raise HTTPException(status_code=400, detail=f"input_dir not found: {p}")
            file_list = _iter_files(p)
        elif paths:
            file_list = []
            for s in paths:
                sp = _resolve_inside_base(_normalize_user_path(s))
                if not sp.is_file():
                    raise HTTPException(status_code=400, detail=f"path not found: {sp}")
                dst = indir / sp.name
                shutil.copyfile(sp, dst)
                file_list.append(dst)
        else:
            if files:
                _write_uploads_to_dir(files, indir)
            if bundle:
                _extract_zip_to_dir(bundle, indir)
            file_list = _iter_files(indir)

        # выбрать бэкенд
        if backend == "nli":
            clf = ZeroShotNLI(nli_model, device=device, hypothesis=hypothesis)
            def classify_one(t): return clf.classify(t, lbls)
        else:
            clf = SimilarityClassifier(sim_model, device=device,
                                       label_prompt=label_prompt, temperature=temperature)
            def classify_one(t): return clf.classify(t, lbls)

        items = []
        for fp in file_list:
            txt = _read_text_from_file(fp)
            probs = classify_one(txt)
            best = max(probs, key=probs.get) if probs else "__uncertain__"
            items.append({
                "file": str(fp),
                "scores": probs,
                "pred_label": best,
                "pred_prob": float(probs.get(best, 0.0))
            })
        return JSONResponse(content={"items": items, "logs": classify_help})
    finally:
        shutil.rmtree(indir, ignore_errors=True)

@app.post("/doc_type", response_model=ClassifyResponse)
async def doc_type(
    files: Optional[List[UploadFile]] = File(default=None),
    bundle: Optional[UploadFile] = File(default=None),
    input_dir: Optional[str] = Form(default=None),
    paths: Optional[List[str]] = Form(default=None),

    labels: Optional[List[str]] = Form(default=None),
    labels_json: Optional[str] = Form(default=None),

    backend: str = Form("sim"),
    device: str = Form("cpu"),
):
    default_labels = ["Счёт","Договор","Заявление","Акт","Накладная",
                      "Письмо","Претензия","Справка","Протокол","Смета"]
    if not labels and not labels_json:
        labels_json = json.dumps(default_labels, ensure_ascii=False)
    return await classify(files=files, bundle=bundle,
                          input_dir=input_dir, paths=paths,
                          labels=labels, labels_json=labels_json,
                          backend=backend, device=device)
