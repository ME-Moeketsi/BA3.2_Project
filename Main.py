"""
Predict-Care AI
Application Entry Point

Loads saved models, runs integration checks, and prints a compact
PASS/FAIL summary. No model training is started here.

Run from the project root:
    python Main.py
"""

from pathlib import Path
import sys

# ----------------------------------------------------------------
# tqdm (with green bars); fall back to plain printing if missing.
# ----------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kw):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])

        def update(self, n=1):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

        @staticmethod
        def write(msg, **kw):
            print(msg)


GREEN = "green"
BAR_KW = dict(
    ncols=60,
    colour=GREEN,
    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
)


class C:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"


def tag(ok):
    return (C.GREEN + "PASSED" + C.RESET) if ok else (C.RED + "FAILED" + C.RESET)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "Models"
DATA_DIRS = [PROJECT_DIR / "data", PROJECT_DIR / "Data"]


# ================================================================
# HELPERS
# ================================================================

def print_heading(title):
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write(title)
    tqdm.write("=" * 60)


def first_existing(*paths):
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def find_dataset(filename):
    for directory in DATA_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


class CheckCounter:
    """Only prints FAIL lines live; collapses PASS lines into a tally."""

    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0

    def report(self, ok, label, detail=""):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            msg = f"FAIL | {label}"
            if detail:
                msg += f" — {detail}"
            tqdm.write(msg)

    def summary(self):
        total = self.passed + self.failed
        if self.failed == 0:
            tqdm.write(f"{self.name}: {self.passed}/{total} passed")
        else:
            tqdm.write(
                f"{self.name}: {self.passed}/{total} passed, "
                f"{self.failed} FAILED"
            )
        return self.failed == 0


# ================================================================
# FILE CHECKS
# ================================================================

def check_files():
    print_heading("FILES")

    expected = {
        "triage model":       [MODEL_DIR / "triage_model.pkl"],
        "patient-flow model": [MODEL_DIR / "patient_flow_model.pkl"],
        "medication-wait":    [MODEL_DIR / "medication_wait_model.pkl"],
        "combined-wait":      [MODEL_DIR / "combined_wait_model.pkl"],
        "total-time":         [MODEL_DIR / "total_time_excluding_med_wait_model.pkl"],
        "DL model":           [MODEL_DIR / "deep_learning_triage_model.keras"],
        "DL preprocessor":    [MODEL_DIR / "deep_learning_triage_preprocessor.pkl"],
        "NLP classifier":     [MODEL_DIR / "ktas_nlp_model.pkl"],
        "NLP vectorizer":     [MODEL_DIR / "ktas_tfidf_vectorizer.pkl"],
        "time-series model":  [MODEL_DIR / "time_series_model.pkl"],
        "nlp_pipeline.py":    [PROJECT_DIR / "old" / "nlp_pipeline.py",
                               PROJECT_DIR / "nlp_pipeline.py",
                               PROJECT_DIR / "NLP" / "nlp_pipeline.py"],
        "deep_learning.py":   [PROJECT_DIR / "deep_learning.py"],
        "healthcare_ai.py":   [PROJECT_DIR / "healthcare_ai.py"],
    }

    statuses = {}
    missing = []

    with tqdm(total=len(expected), desc="Files", **BAR_KW) as bar:
        for label, candidates in expected.items():
            path = first_existing(*candidates)
            exists = path.is_file()
            statuses[label] = exists
            if not exists:
                missing.append(label)
            bar.update(1)

    total = len(expected)
    found = total - len(missing)

    if not missing:
        tqdm.write(f"Files: {found}/{total} found")
    else:
        tqdm.write(
            f"Files: {found}/{total} found — MISSING: {', '.join(missing)}"
        )

    return statuses


# ================================================================
# TRADITIONAL ML
# ================================================================

def run_traditional_ml_checks():
    print_heading("TRADITIONAL ML")

    try:
        import numpy as np
        import pandas as pd
        import healthcare_ai as ai
    except Exception as exc:
        tqdm.write(f"FAIL | import healthcare_ai — {exc}")
        return False

    c = CheckCounter("ML")
    report = c.report

    loaders = [
        ("triage model",      ai.load_triage_model),
        ("doctor-wait model", ai.load_patient_flow_model),
        ("medication-wait",   ai.load_medication_wait_model),
        ("combined-wait",     ai.load_combined_wait_model),
        ("total-time",        ai.load_total_time_excluding_med_wait_model),
    ]
    packages = {}

    with tqdm(total=len(loaders), desc="ML load", **BAR_KW) as bar:
        for label, loader in loaders:
            try:
                packages[label] = loader()
                report(True, f"Load {label}")
            except Exception as exc:
                report(False, f"Load {label}", str(exc))
            bar.update(1)

    triage_path = find_dataset("ktas_cleaned.csv")
    flow_path = find_dataset("patient_flow_cleaned.csv")
    triage_record = None
    flow_record = None

    with tqdm(total=2, desc="ML samples", **BAR_KW) as bar:
        try:
            if triage_path is not None:
                df = pd.read_csv(triage_path)
                feats = packages.get("triage model", {}).get(
                    "feature_columns", getattr(ai, "TRIAGE_FEATURES", [])
                )
                if all(col in df.columns for col in feats):
                    triage_record = df[feats].dropna().iloc[0].to_dict()
            report(triage_record is not None, "Triage sample loaded")
        except Exception as exc:
            report(False, "Triage sample loaded", str(exc))
        bar.update(1)

        try:
            if flow_path is not None:
                df = pd.read_csv(flow_path)
                feats = packages.get("doctor-wait model", {}).get(
                    "feature_columns", getattr(ai, "FLOW_FEATURES", [])
                )
                if all(col in df.columns for col in feats):
                    flow_record = df[feats].dropna().iloc[0].to_dict()
            report(flow_record is not None, "Flow sample loaded")
        except Exception as exc:
            report(False, "Flow sample loaded", str(exc))
        bar.update(1)

    with tqdm(total=1, desc="ML triage", **BAR_KW) as bar:
        if triage_record is not None and "triage model" in packages:
            try:
                result = ai.predict_triage(triage_record, packages["triage model"])
                ok = len(result) == 1 and int(result[0]) in {1, 2, 3, 4, 5}
                report(ok, "Triage prediction KTAS 1-5", f"result={result}")
            except Exception as exc:
                report(False, "Triage prediction KTAS 1-5", str(exc))
        else:
            report(False, "Triage prediction KTAS 1-5", "no sample/model")
        bar.update(1)

    flow_predictors = [
        ("Doctor-wait",     ai.predict_doctor_wait,
         "doctor-wait model"),
        ("Medication-wait", ai.predict_medication_wait,
         "medication-wait"),
        ("Combined-wait",   ai.predict_combined_wait,
         "combined-wait"),
        ("Total-time",      ai.predict_total_time_excluding_med_wait,
         "total-time"),
    ]

    with tqdm(total=len(flow_predictors), desc="ML flow", **BAR_KW) as bar:
        for label, predictor, key in flow_predictors:
            if flow_record is None or key not in packages:
                report(False, f"{label} prediction", "no sample/model")
            else:
                try:
                    r = np.asarray(
                        predictor(flow_record, packages[key])
                    ).reshape(-1)
                    ok = r.size == 1 and np.isfinite(r[0]) and r[0] >= 0
                    report(ok, f"{label} prediction",
                           f"value={np.round(r, 2).tolist()}")
                except Exception as exc:
                    report(False, f"{label} prediction", str(exc))
            bar.update(1)

    if flow_record is not None and "doctor-wait model" in packages:
        cases = []

        m = dict(flow_record); m.pop("ArrivalTime", None)
        cases.append(("Missing feature", m, (ValueError,)))

        m = dict(flow_record); m["ArrivalTime"] = np.nan
        cases.append(("NaN arrival", m, (ValueError,)))

        m = dict(flow_record); m["ArrivalTime"] = np.inf
        cases.append(("Infinite arrival", m, (ValueError,)))

        m = dict(flow_record); m["ArrivalTime"] = -1
        cases.append(("Negative arrival", m, (ValueError,)))

        m = dict(flow_record); m["Priority"] = 99
        cases.append(("Bad priority", m, (ValueError,)))

        m = dict(flow_record); m["Scenario"] = "__UNKNOWN__"
        cases.append(("Unknown scenario", m, (ValueError,)))

        with tqdm(total=len(cases) + 1, desc="ML edge", **BAR_KW) as bar:
            for label, bad, errs in cases:
                try:
                    ai.predict_doctor_wait(bad, packages["doctor-wait model"])
                    report(False, f"Edge: {label}", "accepted invalid input")
                except errs:
                    report(True, f"Edge: {label}")
                except Exception as exc:
                    report(False, f"Edge: {label}",
                           f"{type(exc).__name__}: {exc}")
                bar.update(1)

            try:
                ai.predict_doctor_wait(pd.DataFrame(),
                                       packages["doctor-wait model"])
                report(False, "Edge: empty batch", "accepted empty input")
            except (ValueError, TypeError):
                report(True, "Edge: empty batch")
            bar.update(1)

    if triage_record is not None and "triage model" in packages:
        with tqdm(total=2, desc="ML triage edge", **BAR_KW) as bar:
            bad = dict(triage_record); bad["Age"] = np.inf
            try:
                ai.predict_triage(bad, packages["triage model"])
                report(False, "Edge: inf age", "accepted invalid input")
            except (ValueError, TypeError):
                report(True, "Edge: inf age")
            bar.update(1)

            unknown = dict(triage_record)
            unknown["Chief_complain"] = "__UNSEEN__"
            try:
                r = ai.predict_triage(unknown, packages["triage model"])
                ok = len(r) == 1 and int(r[0]) in {1, 2, 3, 4, 5}
                report(ok, "Edge: unseen complaint", f"result={r}")
            except Exception as exc:
                report(False, "Edge: unseen complaint", str(exc))
            bar.update(1)

    return c.summary()


# ================================================================
# NLP
# ================================================================

def run_nlp_checks():
    print_heading("NLP")

    try:
        from old.nlp_pipeline import load_nlp_model, predict_ktas
    except Exception as exc:
        tqdm.write(f"FAIL | NLP import — {exc}")
        return False

    c = CheckCounter("NLP")

    with tqdm(total=1, desc="NLP load", **BAR_KW) as bar:
        try:
            load_nlp_model()
            c.report(True, "NLP model load")
        except Exception as exc:
            c.report(False, "NLP model load", str(exc))
        bar.update(1)

    with tqdm(total=2, desc="NLP checks", **BAR_KW) as bar:
        try:
            prediction = predict_ktas("chest pain")
            level = int(prediction["predicted_ktas"])
            ok = level in {1, 2, 3, 4, 5}
            c.report(ok, "NLP prediction KTAS 1-5",
                     f"level={level}, cat={prediction.get('emergency_category')}")
        except Exception as exc:
            c.report(False, "NLP prediction KTAS 1-5", str(exc))
        bar.update(1)

        try:
            predict_ktas("   ")
            c.report(False, "Reject blank complaint", "accepted blank input")
        except (ValueError, TypeError):
            c.report(True, "Reject blank complaint")
        except Exception as exc:
            c.report(False, "Reject blank complaint", str(exc))
        bar.update(1)

    return c.summary()


# ================================================================
# DEEP LEARNING
# ================================================================

def run_deep_learning_checks():
    print_heading("DEEP LEARNING")

    try:
        import numpy as np
        import pandas as pd
        from deep_learning import (
            TRIAGE_FEATURES, load_deep_learning_model, predict_ktas,
        )
    except Exception as exc:
        tqdm.write(f"FAIL | DL import — {exc}")
        return False

    c = CheckCounter("DL")

    with tqdm(total=1, desc="DL load", **BAR_KW) as bar:
        try:
            model, pre, _ = load_deep_learning_model()
            c.report(True, "DL model load")
        except Exception as exc:
            c.report(False, "DL model load", str(exc))
            bar.update(1)
            return c.summary()
        bar.update(1)

    try:
        path = find_dataset("ktas_cleaned.csv")
        if path is None:
            raise FileNotFoundError("KTAS dataset not found")
        data = pd.read_csv(path)
        usable = data[TRIAGE_FEATURES].dropna()
        if usable.empty:
            raise ValueError("no complete records")
    except Exception as exc:
        c.report(False, "Load DL sample", str(exc))
        return c.summary()

    sample = usable.iloc[[0]].copy()
    probs = None

    with tqdm(total=6, desc="DL checks", **BAR_KW) as bar:
        try:
            preds, probs = predict_ktas(model, pre, sample)
            preds = np.asarray(preds).reshape(-1)
            probs = np.asarray(probs)
            ok = preds.size == 1 and int(preds[0]) in {1, 2, 3, 4, 5}
            c.report(ok, "DL prediction KTAS 1-5",
                     f"result={preds.tolist()}")
        except Exception as exc:
            probs = None
            c.report(False, "DL prediction KTAS 1-5", str(exc))
        bar.update(1)

        if probs is not None:
            shape_ok = probs.shape == (1, 5)
            c.report(shape_ok, "Probabilities shape (1,5)",
                     f"shape={probs.shape}")
            prob_ok = (
                shape_ok
                and np.isfinite(probs).all()
                and np.all(probs >= 0) and np.all(probs <= 1)
                and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            )
            c.report(prob_ok, "Probabilities sum to 1",
                     f"sum={probs[0].sum():.4f}" if shape_ok else "")
        bar.update(1)

        batch = usable.iloc[:2].copy()
        if len(batch) == 2:
            try:
                bp, bprob = predict_ktas(model, pre, batch)
                bp = np.asarray(bp).reshape(-1)
                bprob = np.asarray(bprob)
                ok = (
                    bp.shape == (2,)
                    and all(int(v) in {1, 2, 3, 4, 5} for v in bp)
                    and bprob.shape == (2, 5)
                )
                c.report(ok, "Batch of 2 patients", f"pred={bp.tolist()}")
            except Exception as exc:
                c.report(False, "Batch of 2 patients", str(exc))
        else:
            c.report(False, "Batch of 2 patients", "not enough records")
        bar.update(1)

        incomplete = sample.drop(columns=[TRIAGE_FEATURES[0]])
        try:
            predict_ktas(model, pre, incomplete)
            c.report(False, "Reject missing feature",
                     "accepted invalid input")
        except ValueError:
            c.report(True, "Reject missing feature")
        except Exception as exc:
            c.report(False, "Reject missing feature", str(exc))
        bar.update(1)

        try:
            predict_ktas(model, pre, sample.iloc[0:0])
            c.report(False, "Reject empty batch", "accepted empty input")
        except (ValueError, TypeError):
            c.report(True, "Reject empty batch")
        except Exception as exc:
            c.report(False, "Reject empty batch", str(exc))
        bar.update(1)

    return c.summary()


# ================================================================
# MAIN
# ================================================================

def main():
    print_heading("PREDICT-CARE AI — STARTUP")
    tqdm.write(f"Project: {PROJECT_DIR}")
    tqdm.write(f"Python:  {sys.executable}")

    check_files()
    ml_ok  = run_traditional_ml_checks()
    nlp_ok = run_nlp_checks()
    dl_ok  = run_deep_learning_checks()

    print_heading("SUMMARY")
    tqdm.write(f"Traditional ML : {tag(ml_ok)}")
    tqdm.write(f"NLP            : {tag(nlp_ok)}")
    tqdm.write(f"Deep Learning  : {tag(dl_ok)}")
    tqdm.write("No training started. Prototype only — not clinical advice.")

    return 0 if ml_ok and nlp_ok and dl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())