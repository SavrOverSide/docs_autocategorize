#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автокатегоризация/кластеризация + предсказание.
- fit: читает папку/файл, извлекает текст, эмбеддинги, выбирает k, KMeans, (опц.) рекурсивный рекластеринг, (опц.) пост-merge.
        Пишет categories.json (описание кластеров), assignments.jsonl и модель центроидов.
        По флагу --predict-on-fit сразу выдаёт predictions.jsonl + doc_types.txt (zero-shot тип документа).
- predict: загружает centroids и для каждого файла выдаёт:
        pred_label, sim_to_centroid, tags и главное — scores: распределение вероятностей по кластерам (softmax от косинусов).
        Параллельно делает zero-shot тип документа: doc_type_pred + doc_type_scores.
"""

import argparse
import json
import os
import re
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from sentence_transformers import SentenceTransformer
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# optional deps
try:
    import pymorphy2
except Exception:
    pymorphy2 = None

try:
    import yake
except Exception:
    yake = None

try:
    from keybert import KeyBERT
except Exception:
    KeyBERT = None


# ---------------- IO & Preproc ----------------

READ_TXT_EXT = {".txt", ".md", ".csv", ".tsv", ".xml", ".html", ".htm", ".json", ".rtf"}
READ_PDF_EXT = {".pdf"}

@dataclass
class DocItem:
    name: str
    path: str
    raw: str
    clean: str

def is_hidden(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)

def read_files(root: Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    out = []
    for p in root.rglob("*"):
        if p.is_file() and not is_hidden(p):
            out.append(p)
    return sorted(out)

def extract_any_text(p: Path, max_chars: int = 20000) -> str:
    suf = p.suffix.lower()
    try:
        if suf in READ_TXT_EXT:
            t = p.read_text(encoding="utf-8", errors="ignore")
            return t[:max_chars]
        elif suf in READ_PDF_EXT and pdf_extract_text is not None:
            t = pdf_extract_text(str(p)) or ""
            return t[:max_chars]
        else:
            # браво, если бинарь — вернём пусто
            return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""

_non_word = re.compile(r"[^A-Za-zА-Яа-я0-9ёЁ]+", re.U)

def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_and_lemmatize(t: str, stopwords: set) -> List[str]:
    tokens = [w.lower() for w in _non_word.split(t) if w.strip()]
    if pymorphy2 is not None:
        morph = pymorphy2.MorphAnalyzer()
        norm = []
        for w in tokens:
            if re.search(r"[A-Za-z]", w):  # англ не трогаем pymorphy2
                if w not in stopwords:
                    norm.append(w)
                continue
            try:
                parse = morph.parse(w)[0]
                lemma = parse.normal_form
                if lemma and lemma not in stopwords:
                    norm.append(lemma)
            except Exception:
                if w not in stopwords:
                    norm.append(w)
        return norm
    # без pymorphy — просто фильтр стопов
    return [w for w in tokens if w not in stopwords]

def tokens_to_text(tokens: List[str]) -> str:
    return " ".join(tokens)

def embed_texts(texts: List[str], model_name: str, device: str = "cpu",
                batch_size: int = 32, max_seq_length: int = 512) -> Tuple[np.ndarray, Dict]:
    model = SentenceTransformer(model_name, device=device)
    try:
        # не у всех моделей это поле есть
        model.max_seq_length = max_seq_length
    except Exception:
        pass
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    meta = {"model": model_name}
    return emb, meta


# ---------------- Labeling ----------------

def ctfidf_top_terms(texts_per_cluster: List[List[str]], top_k: int = 10) -> List[List[str]]:
    # объединяем тексты кластера в один документ
    docs = [" ".join(xs) if isinstance(xs, list) else str(xs) for xs in texts_per_cluster]
    if not any(docs):
        return [[] for _ in docs]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0,
                          smooth_idf=True, sublinear_tf=True)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    tops: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i).toarray().ravel()
        if row.sum() == 0:
            tops.append([])
            continue
        idx = np.argsort(row)[::-1][:top_k]
        tops.append([terms[j] for j in idx])
    return tops

def yake_top_terms(texts_per_cluster: List[List[str]], top_k: int = 10, lang: str = "ru") -> List[List[str]]:
    if yake is None:
        return [[] for _ in texts_per_cluster]
    tops: List[List[str]] = []
    for xs in texts_per_cluster:
        text = " ".join(xs)[:20000]
        try:
            kw = yake.KeywordExtractor(lan=lang, top=top_k)
            pairs = kw.extract_keywords(text)
            pairs = sorted(pairs, key=lambda x: x[1])
            tops.append([k for k, _ in pairs[:top_k]])
        except Exception:
            tops.append([])
    return tops

def build_keybert(model_name: str, device: str = "cpu"):
    if KeyBERT is None:
        return None
    m = SentenceTransformer(model_name, device=device)
    return KeyBERT(model=m)

def extract_tags_with_backend(text: str, backend: str, kb,
                              top_n: int = 8, stopwords: set = None,
                              **kwargs) -> List[str]:
    stopwords = stopwords or set()
    if backend == "yake" and yake is not None:
        try:
            kw = yake.KeywordExtractor(lan="ru", top=top_n)
            pairs = kw.extract_keywords(text[:20000])
            pairs = sorted(pairs, key=lambda x: x[1])
            return [k for k, _ in pairs[:top_n] if k not in stopwords]
        except Exception:
            return []
    if backend == "keybert" and kb is not None:
        try:
            pairs = kb.extract_keywords(text[:20000], top_n=top_n)
            return [k for k, _ in pairs if k not in stopwords]
        except Exception:
            return []
    return []

def truncate_for_tags(text: str, max_len: int = 8000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len]


def pick_k(emb: np.ndarray, k_min: int, k_max: int, tol: float = 0.25) -> int:
    """Выбор k по силуэту с приоритетом меньшего k в пределах tol к лучшему."""
    n = emb.shape[0]
    if n <= 1:
        return 1
    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    k_max = min(k_max, n)  # не больше числа документов
    best_k, best_s = k_min, -1.0
    scores: List[Tuple[int, float]] = []
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
            if len(set(km.labels_)) < 2:
                s = -1.0
            else:
                s = float(silhouette_score(emb, km.labels_))
        except Exception:
            s = -1.0
        scores.append((k, s))
        if s > best_s:
            best_s, best_k = s, k
    # берем наименьший k, чей score >= best_s * (1 - tol)
    threshold = best_s * (1 - max(0.0, tol))
    candidates = [k for k, s in scores if s >= threshold]
    return min(candidates) if candidates else best_k

def recluster_cluster(emb: np.ndarray, idxs: List[int],
                      min_k: int = 2, max_k: int = 4,
                      child_min_size: int = 2,
                      tol: float = 0.05,
                      max_dominance: float = 0.90) -> Optional[np.ndarray]:
    """Рекластеринг подмассива idxs. Возвращает массив длины N с -1 вне idxs и локальными метками в idxs."""
    if len(idxs) < max(min_k, child_min_size):
        return None
    sub = emb[idxs]
    k2 = pick_k(sub, min_k, max_k, tol=tol)
    km = KMeans(n_clusters=k2, n_init=10, random_state=1337).fit(sub)
    labs = km.labels_
    # dominance check
    sizes = [(c, int((labs == c).sum())) for c in range(k2)]
    sizes.sort(key=lambda x: x[1], reverse=True)
    if sizes[0][1] / len(idxs) >= max_dominance:
        return None
    # фильтр мелочи
    valid = {c for c, sz in sizes if sz >= child_min_size}
    if len(valid) <= 1:
        return None
    out = np.full(emb.shape[0], -1, dtype=int)
    for local_idx, gi in enumerate(idxs):
        out[gi] = labs[local_idx] if labs[local_idx] in valid else -1
    return out

def choose_label(ct_terms: List[str], yk_terms: List[str], df_stop: set, local_topk: int = 2) -> Tuple[str, List[str]]:
    ranked = []
    for w in (ct_terms or []):
        if w and w not in df_stop:
            ranked.append(w)
    for w in (yk_terms or []):
        if w and (w not in df_stop) and (w not in ranked):
            ranked.append(w)
    if not ranked:
        return ("Без названия", [])
    label = ", ".join(ranked[:max(1, local_topk)])
    alt = ranked[:10]
    return (label, alt)


# ---------------- Doc type zero-shot ----------------

DEFAULT_DOC_TYPES = [
    "счет", "счет-фактура", "договор", "заявление",
    "накладная", "акт", "спецификация", "протокол"
]

def parse_doc_types(arg: str) -> List[str]:
    if arg is None:
        return []
    arg = str(arg).strip()
    if arg == "":
        return DEFAULT_DOC_TYPES
    p = Path(arg)
    if p.exists():
        try:
            if p.suffix.lower() == ".json":
                return json.loads(p.read_text(encoding="utf-8"))
            lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
            return [x for x in lines if x]
        except Exception:
            pass
    return [x.strip() for x in arg.split(",") if x.strip()]

def make_type_prompts(types: List[str]) -> List[str]:
    hints = {
        "счет": ["счет на оплату", "invoice", "инвойс"],
        "счет-фактура": ["НДС", "invoice factura", "счет-фактура"],
        "договор": ["contract", "agreement", "обязательства", "стороны"],
        "заявление": ["форма заявления", "application", "прошу"],
        "накладная": ["товарная накладная", "Торг-12", "waybill", "отгрузка"],
        "акт": ["акт выполненных работ", "оказанных услуг", "acceptance act"],
        "спецификация": ["перечень позиций", "характеристики", "specification"],
        "протокол": ["протокол заседания", "meeting minutes"]
    }
    prompts = []
    for t in types:
        hs = " ".join(hints.get(t.lower(), []))
        prompts.append(f"Тип документа: {t}. Ключевые слова: {hs}")
    return prompts


# ---------------- FIT ----------------

def run_fit(args):
    root = Path(args.input)
    paths = read_files(root)
    if not paths:
        raise SystemExit("No files found.")

    # базовый стоп-лист (минимальный)
    base_stop = {"год","г.","стать","рис","табл","данный","можно","также","свой","который",
                 "the","and","for","with","from","this","that","have","has","was","are"}
    stopwords = base_stop

    docs: List[DocItem] = []
    for p in paths:
        raw = extract_any_text(p)
        clean = clean_text(raw)
        toks = normalize_and_lemmatize(clean, stopwords)
        clean2 = tokens_to_text(toks)
        if not clean2:
            continue
        docs.append(DocItem(name=p.name, path=str(p), raw=raw, clean=clean2))
    if not docs:
        raise SystemExit("No readable documents.")

    texts = [d.clean for d in docs]
    emb, emb_meta = embed_texts(texts, args.model_name, device=args.device,
                                batch_size=args.batch_size, max_seq_length=args.max_seq_length)
    emb = sk_normalize(emb)

    # --- выбор k и меток с учётом малых выборок ---
    n_docs = len(docs)
    if n_docs == 1:
        k = 1
        labels = np.zeros(n_docs, dtype=int)
        print(f"[k-select] docs={n_docs}, k={k}", flush=True)
    else:
        if args.k is not None and args.k > 1:
            k = min(int(args.k), n_docs)  # не больше числа документов
        else:
            k = pick_k(emb, args.min_k, args.max_k, tol=args.k_select_tol)
            k = min(max(2, int(k)), n_docs)
        print(f"[k-select] docs={n_docs}, k={k}", flush=True)
        # KMeans только если k>1
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
        labels = km.labels_.astype(int)

    # рекурсивный доразрез крупных кластеров
    if args.recluster_large and k > 1:
        depth = 0
        while depth < args.recluster_max_depth:
            depth += 1
            changed = False
            for c in range(int(k)):
                idxs = [i for i, lab in enumerate(labels) if int(lab) == c]
                if len(idxs) < args.recluster_min_size:
                    continue
                child = recluster_cluster(
                    emb, idxs,
                    min_k=args.recluster_min_k, max_k=args.recluster_max_k,
                    child_min_size=args.recluster_child_min_size,
                    tol=args.recluster_tol, max_dominance=args.recluster_max_dominance
                )
                if child is None:
                    continue
                # перенумеруем деток поверх глобальной разметки
                max_lab = int(labels.max())
                child_ids = sorted(set(int(x) for x in child if x >= 0))
                mapping = {cid: (max_lab + 1 + j) for j, cid in enumerate(child_ids)}
                for i in range(len(labels)):
                    if int(child[i]) >= 0:
                        labels[i] = int(mapping[int(child[i])])
                changed = True
                k = int(labels.max()) + 1
            if not changed:
                break

    # пост-merge маленьких кластеров
    if args.merge_small and k > 1:
        for _ in range(args.merge_small_iter):
            K = int(labels.max()) + 1
            cents = np.vstack([
                emb[labels == ci].mean(axis=0) if (labels == ci).sum() > 0
                else np.zeros((emb.shape[1],))
                for ci in range(K)
            ])
            cents = sk_normalize(cents)
            sizes = [(ci, int((labels == ci).sum())) for ci in range(K)]
            small = [ci for ci, sz in sizes if sz <= args.merge_small_max_size]
            big   = [ci for ci, sz in sizes if sz > args.merge_small_max_size]
            if not small or not big:
                break
            for s in small:
                if (labels == s).sum() == 0:
                    continue
                sims = cents[big] @ cents[s]
                j = int(np.argmax(sims))
                if float(sims[j]) >= args.merge_small_sim_threshold:
                    target = int(big[j])
                    labels[labels == s] = target
        # уплотним метки
        uniq = sorted(set(int(x) for x in labels))
        remap = {old: i for i, old in enumerate(uniq)}
        labels = np.array([remap[int(x)] for x in labels], dtype=int)

    K = int(labels.max()) + 1
    clusters = {i: [] for i in range(K)}
    for i, lab in enumerate(labels):
        clusters[int(lab)].append(i)

    # тексты по кластерам
    texts_per_cluster = [[texts[i] for i in clusters[c]] for c in range(K)]

    # топ-термы
    ct_terms = ctfidf_top_terms(texts_per_cluster, top_k=args.label_topn)
    yk_terms = yake_top_terms(texts_per_cluster, top_k=args.label_topn, lang="ru") if args.label_use_yake else [[] for _ in range(K)]

    # DF stoplist (по корпусу)
    df = {}
    for t in texts:
        seen = set(t.split())
        for w in seen:
            df[w] = df.get(w, 0) + 1
    df_ratio = {w: df[w]/len(texts) for w in df}
    df_stop = {w for w, r in df_ratio.items() if r >= args.df_stop_threshold}

    categories = {
        "library": "kmeans+recluster+postmerge" if (args.recluster_large and k > 1) or (args.merge_small and k > 1) else "kmeans",
        "embedding_model": args.model_name,
        "k": int(K),
        "labeling": {
            "method": "ctfidf+centroid+yake",
            "df_stop_threshold": args.df_stop_threshold,
            "label_topn": int(args.label_topn),
            "label_local_stop_topk": int(args.label_local_stop_topk),
            "use_yake": bool(args.label_use_yake)
        },
        "recluster": {
            "enabled": bool(args.recluster_large and k > 1),
            "max_depth": int(args.recluster_max_depth),
            "min_size": int(args.recluster_min_size),
            "child_min_size": int(args.recluster_child_min_size),
            "min_k": int(args.recluster_min_k),
            "max_k": int(args.recluster_max_k),
            "tol": float(args.recluster_tol),
            "max_dominance": float(args.recluster_max_dominance)
        },
        "post_merge": {
            "enabled": bool(args.merge_small and k > 1),
            "max_size": int(args.merge_small_max_size),
            "sim_threshold": float(args.merge_small_sim_threshold),
            "iter": int(args.merge_small_iter)
        },
        "clusters": []
    }

    # финальные метки
    for c in range(K):
        label, alt = choose_label(ct_terms[c], yk_terms[c], df_stop, args.label_local_stop_topk)
        examples = [docs[i].name for i in clusters[c][:3]]
        categories["clusters"].append({
            "id": int(c),
            "label": label,
            "alt_labels": alt,
            "size": int(len(clusters[c])),
            "top_terms": list(dict.fromkeys((ct_terms[c] or []) + (yk_terms[c] or [])))[:10],
            "examples": examples
        })

    # assignments.jsonl
    with Path(args.out_assign).open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({
                "name": d.name, "path": d.path,
                "cluster": int(labels[i]),
                "label": categories["clusters"][int(labels[i])]["label"]
            }, ensure_ascii=False) + "\n")

    Path(args.out_cats).write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")

    # centroids model
    cents = np.vstack([
        emb[labels == ci].mean(axis=0) if (labels == ci).sum() > 0 else np.zeros((emb.shape[1],))
        for ci in range(K)
    ])
    cents = sk_normalize(cents)
    model = {
        "embedding": emb_meta,
        "centroids": cents.tolist(),
        "clusters": categories["clusters"],
        "df_stop_threshold": float(args.df_stop_threshold)
    }
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_model).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ wrote {args.out_cats}, {args.out_assign}, and model {args.save_model}")

    # one-shot predict прямо из fit
    if getattr(args, "predict_on_fit", False):
        from argparse import Namespace
        pred_args = Namespace(
            input=args.input,
            model=args.save_model,
            device=args.device,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            sim_threshold=None,
            prob_temperature=args.prob_temperature,
            tags_backend=args.tags_backend if hasattr(args, "tags_backend") else "none",
            tags_device=args.device,
            tags_top_n=args.tags_top_n if hasattr(args, "tags_top_n") else 8,
            out=args.predict_out,
            doc_types=args.doc_types,
            doc_types_temperature=args.doc_types_temperature,
            doc_types_out=args.doc_types_out
        )
        run_predict(pred_args)


# ---------------- PREDICT ----------------

def _softmax_row(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x / max(1e-8, temp)) - np.max(x)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))

def run_predict(args):
    mpath = Path(args.model)
    if not mpath.exists():
        raise SystemExit(f"Model not found: {mpath}")
    model = json.loads(mpath.read_text(encoding="utf-8"))
    centroids = np.asarray(model["centroids"], dtype=np.float32)
    centroids = sk_normalize(centroids)
    labels_list = [c.get("label", f"Класс {i}") for i, c in enumerate(model.get("clusters", []))]

    root = Path(args.input)
    paths = read_files(root)
    if not paths:
        raise SystemExit("No files found.")

    # стоп слова (минимум)
    base_stop = {"год","г.","стать","рис","табл","данный","можно","также","свой","который",
                 "the","and","for","with","from","this","that","have","has","was","are"}

    docs: List[DocItem] = []
    for p in paths:
        raw = extract_any_text(p)
        clean = clean_text(raw)
        toks = normalize_and_lemmatize(clean, base_stop)
        clean2 = tokens_to_text(toks)
        docs.append(DocItem(name=p.name, path=str(p), raw=raw, clean=clean2))

    texts = [d.clean for d in docs]
    emb_model_name = model.get("embedding", {}).get("model", args.model_name)
    embs, _ = embed_texts(texts, emb_model_name, device=args.device,
                          batch_size=args.batch_size, max_seq_length=args.max_seq_length)
    embs = sk_normalize(embs)

    sims = embs @ centroids.T  # [N, K]
    probs = np.vstack([_softmax_row(sims[i], temp=args.prob_temperature) for i in range(sims.shape[0])])

    kb = None
    if getattr(args, "tags_backend", "none") == "keybert":
        kb = build_keybert(emb_model_name, args.tags_device)

    # ---- zero-shot doc types ----
    dt_list = parse_doc_types(getattr(args, "doc_types", ""))
    probs_dt = None
    if dt_list:
        type_prompts = make_type_prompts(dt_list)
        type_embs, _ = embed_texts(type_prompts, emb_model_name, device=args.device,
                                   batch_size=max(8, len(type_prompts)), max_seq_length=args.max_seq_length)
        type_embs = sk_normalize(type_embs)
        sims_dt = embs @ type_embs.T
        probs_dt = np.vstack([_softmax_row(sims_dt[i], temp=getattr(args, "doc_types_temperature", 0.7))
                              for i in range(sims_dt.shape[0])])
        dto = getattr(args, "doc_types_out", "")
        if dto:
            Path(dto).write_text("\n".join(dt_list) + "\n", encoding="utf-8")

    with Path(args.out).open("w", encoding="utf-8") as f:
        total = len(docs)
        for i, d in enumerate(docs, start=1):
            row_sim = sims[i-1]
            row_prob = probs[i-1]
            cid = int(np.argmax(row_prob))
            sc = float(row_sim[cid])
            plabel = labels_list[cid] if cid < len(labels_list) else f"Класс {cid}"

            if getattr(args, "sim_threshold", None) is not None and sc < args.sim_threshold:
                plabel = "Прочее"

            tags = []
            if getattr(args, "tags_backend", "none") != "none":
                tags = extract_tags_with_backend(
                    truncate_for_tags(d.clean), args.tags_backend, kb,
                    top_n=getattr(args, "tags_top_n", 8), stopwords=set()
                )

            score_map = { (labels_list[j] if j < len(labels_list) else f"Класс {j}"): float(row_prob[j])
                          for j in range(len(row_prob)) }

            rec = {
                "name": d.name, "path": d.path,
                "pred_cluster": cid, "pred_label": plabel,
                "sim_to_centroid": round(sc, 4),
                "scores": score_map,
                "tags": tags
            }
            if probs_dt is not None:
                row_dt = probs_dt[i-1]
                j = int(np.argmax(row_dt))
                rec["doc_type_pred"] = dt_list[j]
                rec["doc_type_scores"] = {dt_list[k]: float(row_dt[k]) for k in range(len(dt_list))}

            print(f"[predict] {i}/{total} {d.name} -> {plabel} (sim={sc:.3f})", flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ wrote {args.out}")


# ---------------- CLI ----------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Auto categorize documents (fit/predict)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fit
    f = sub.add_parser("fit", help="Train/cluster on a folder or file")
    f.add_argument("input", help="File or directory")
    f.add_argument("--device", default="cpu")
    f.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--max-seq-length", type=int, default=512)

    f.add_argument("--k", type=int, default=None)
    f.add_argument("--min-k", type=int, default=7)
    f.add_argument("--max-k", type=int, default=12)
    f.add_argument("--k-select-tol", type=float, default=0.25)

    f.add_argument("--recluster-large", action="store_true")
    f.add_argument("--recluster-max-depth", type=int, default=1)
    f.add_argument("--recluster-min-size", type=int, default=6)
    f.add_argument("--recluster-child-min-size", type=int, default=2)
    f.add_argument("--recluster-min-k", type=int, default=2)
    f.add_argument("--recluster-max-k", type=int, default=4)
    f.add_argument("--recluster-tol", type=float, default=0.05)
    f.add_argument("--recluster-max-dominance", type=float, default=0.90)

    f.add_argument("--merge-small", action="store_true")
    f.add_argument("--merge-small-max-size", type=int, default=2)
    f.add_argument("--merge-small-sim-threshold", type=float, default=0.55)
    f.add_argument("--merge-small-iter", type=int, default=1)

    f.add_argument("--df-stop-threshold", type=float, default=0.6)
    f.add_argument("--label-topn", type=int, default=8)
    f.add_argument("--label-local-stop-topk", type=int, default=2)
    f.add_argument("--label-use-yake", action="store_true")

    # чтобы fit не ругался, если прокинете теги (используются только в predict/on-fit)
    f.add_argument("--tags-backend", choices=["none", "yake", "keybert"], default="none")
    f.add_argument("--tags-top-n", type=int, default=8)

    f.add_argument("--out-cats", default="categories.json")
    f.add_argument("--out-assign", default="assignments.jsonl")
    f.add_argument("--save-model", default="models/semantic_centroids.json")

    # one-shot предсказание сразу из fit
    f.add_argument("--predict-on-fit", action="store_true", help="Сразу после fit запускает predict на том же input")
    f.add_argument("--predict-out", default="predictions.jsonl", help="Куда писать предсказания при --predict-on-fit")
    f.add_argument("--prob-temperature", type=float, default=0.8, help="Температура softmax для распределения по кластерам")
    f.add_argument("--doc-types", default="", help="Список типов через запятую или путь к .json/.txt; пусто = дефолтный список")
    f.add_argument("--doc-types-temperature", type=float, default=0.7, help="Температура softmax для типов документов")
    f.add_argument("--doc-types-out", default="doc_types.txt", help="Куда сохранить список типов (если включен one-shot)")

    # predict
    p = sub.add_parser("predict", help="Predict cluster for new docs")
    p.add_argument("input", help="File or directory")
    p.add_argument("--model", required=True, help="Path to saved centroids model (JSON)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-name", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--sim-threshold", type=float, default=None, help="optional: mark as 'Прочее' if sim below")
    p.add_argument("--prob-temperature", type=float, default=1.0, help="softmax temperature over centroid sims")

    p.add_argument("--tags-backend", choices=["none","yake","keybert"], default="none")
    p.add_argument("--tags-device", default="cpu")
    p.add_argument("--tags-top-n", type=int, default=8)

    # doc-type zero-shot
    p.add_argument("--doc-types", default="", help="Список типов через запятую или путь к .json/.txt; пусто = дефолт")
    p.add_argument("--doc-types-temperature", type=float, default=0.7)
    p.add_argument("--doc-types-out", default="", help="Если задан, сохранит список типов сюда")

    p.add_argument("--out", default="predictions.jsonl")

    return ap

def main():
    args = build_argparser().parse_args()
    if args.cmd == "fit":
        run_fit(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
