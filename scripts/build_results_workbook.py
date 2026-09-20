"""Builds a workbook of the text channel's results.

Five sheets: an index, what each experiment did, the best arm per
experiment, the declared contrasts, and every logged run.

    python scripts/build_results_workbook.py

Every number is on the same 48 held-out participants, and every arm is a
single run at seed 42. That last point governs how the tables should be
read and is repeated wherever a ranking appears.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, logging_utils, splits  # noqa: E402

OUTPUT = PROJECT_DIR / "results" / "exports" / f"resultados_texto_{date.today():%Y%m%d}.xlsx"

TARGETS = ["PHQ9", "GAD7"]

# What each notebook set out to measure. Keyed by the combination of
# notebook, family, granularity and augmentation that identifies its rows.
EXPERIMENTS = {
    ("1", "frozen", "participant", "no_augmentation"): (
        "E0-E2 congelados, nivel participante",
        "TF-IDF, BETO y multilingual-e5 en tres tamanos. El encoder no se entrena: "
        "emite un vector por participante (promedio de sus respuestas) y nueve "
        "regresores clasicos compiten por validacion cruzada agrupada.",
        "Establece la linea base de cada representacion sin que el transformer "
        "aprenda nada de la tarea."),
    ("1", "frozen", "participant", "augmented"): (
        "E0-E2 congelados + aumentacion LLM",
        "Lo mismo, con una parafrasis generada por LLM por cada participante de "
        "entrenamiento. 110 personas pasan a 220 filas, agrupadas por participante "
        "original para que parafrasis y fuente nunca crucen un fold.",
        "Mide si duplicar las filas de entrenamiento con parafrasis ayuda."),
    ("2", "frozen", "question", "no_augmentation"): (
        "E0-E2 congelados, nivel pregunta",
        "Cada respuesta es una fila con la etiqueta de su participante replicada. "
        "376 filas en vez de 110. Las predicciones se promedian por participante "
        "antes de cualquier metrica.",
        "Mide si entrenar sobre respuestas sueltas supera a promediar los "
        "embeddings antes de entrenar."),
    ("2", "frozen", "question", "augmented"): (
        "E0-E2 congelados, nivel pregunta + aumentacion",
        "Nivel pregunta con parafrasis: 752 filas de entrenamiento sobre las mismas "
        "110 personas.",
        "Cruza las dos manipulaciones anteriores."),
    ("3", "frozen", "participant", "no_augmentation2"): (
        "E3 adaptacion de dominio con TSDAE",
        "multilingual-e5-small adaptado sin supervision a las transcripciones del "
        "propio corpus, evaluado en cuatro presupuestos de adaptacion (5, 10, 20 y "
        "30 epocas) y congelado despues.",
        "Mide si adaptar el encoder al vocabulario del corpus mejora la "
        "representacion. Los cuatro presupuestos se reportan como curva, no se "
        "elige el mejor."),
    ("4", "fine-tuning", "question", "augmented"): (
        "E5 fine-tuning con holdout unico",
        "Siete checkpoints afinados de extremo a extremo con una cabeza de "
        "regresion: e5 en tres tamanos, BETO y BERTIN (solo espanol), mDeBERTa-v3 y "
        "XLM-R (multilingues). El corte de epocas se decide con un 20 por ciento de "
        "los participantes de entrenamiento, apartados una sola vez.",
        "Mide si dejar que el encoder aprenda la tarea supera a congelarlo, y "
        "contrasta preentrenamiento monolingue frente a multilingue y la "
        "arquitectura de mDeBERTa frente a XLM-R."),
    ("5", "fine-tuning-cv", "question", "augmented"): (
        "E6 fine-tuning con validacion cruzada, aumentado",
        "Los mismos siete checkpoints, pero el presupuesto de epocas sale de una "
        "validacion cruzada de 5 folds y el modelo evaluado se reajusta sobre los "
        "110 participantes completos.",
        "Mide cuanto costaba el holdout unico: alli 22 personas nunca aportaban "
        "gradientes y la senal de parada descansaba en esas mismas 22."),
    ("5", "fine-tuning-cv", "question", "no_augmentation"): (
        "E6 fine-tuning con validacion cruzada, sin aumentar",
        "Igual que el anterior, sin parafrasis.",
        "Aisla el efecto de la aumentacion dentro del esquema con validacion "
        "cruzada."),
}


def readable_checkpoint(value: str) -> str:
    """The TSDAE arms record an absolute path to a local checkpoint, which is
    unreadable in a cell and meaningless on another machine."""
    text = str(value).replace("\\", "/")
    if "tsdae" in text:
        # Keep the adaptation run's directory: several runs share ep<NN> names.
        return "/".join(text.rstrip("/").split("/")[-2:])
    return text


def predictor(family: str, selected: str) -> str:
    """What produced the number: a classical regressor chosen by
    cross-validation, or the fine-tuned encoder itself."""
    if family == "fine-tuning":
        return "fine-tuning de extremo a extremo (holdout unico)"
    if family == "fine-tuning-cv":
        return "fine-tuning de extremo a extremo (validacion cruzada)"
    return str(selected)


def load() -> pd.DataFrame:
    runs = logging_utils.load_all_runs()
    # Notebook 3 shares its family, granularity and augmentation with notebook
    # 1, so the notebook number is what separates them.
    runs["key"] = list(zip(runs.notebook.astype(str), runs.family.astype(str),
                            runs.granularity.astype(str),
                            np.where(runs.notebook.astype(str) == "3",
                                      "no_augmentation2", runs.augmentation.astype(str))))
    labels = {k: v[0] for k, v in EXPERIMENTS.items()}
    runs["Experimento"] = [labels.get(k, " | ".join(k)) for k in runs["key"]]
    runs["Encoder"] = runs.encoder_model.map(readable_checkpoint)
    runs["Predictor"] = [predictor(f, m) for f, m
                          in zip(runs.family.astype(str), runs.selected_model)]
    return runs


def baseline_row() -> dict:
    """Predicting the training mean. Every arm is read against this."""
    frame = data.load_dataset()
    train_ids, test_ids = splits.train_test_participants(frame)
    out = {}
    for target, column in config.TARGET_COLUMNS.items():
        mean = frame.loc[train_ids, column].mean()
        y = frame.loc[test_ids, column].to_numpy()
        out[target] = {
            "rmse": float(np.sqrt(((y - mean) ** 2).mean())),
            "mae": float(np.abs(y - mean).mean()),
            "r2": float(1 - ((y - mean) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        }
    return out


def experiments_sheet(runs: pd.DataFrame) -> pd.DataFrame:
    counts = runs.groupby("Experimento")["encoder"].nunique()
    rows = []
    for key, (label, what, why) in EXPERIMENTS.items():
        rows.append({
            "Notebook": key[0], "Experimento": label,
            "Que se hizo": what, "Que mide": why,
            "Brazos": int(counts.get(label, 0)),
            "Granularidad": "participante" if key[2] == "participant" else "pregunta",
            "Aumentacion": "no" if key[3].startswith("no_augmentation") else "si",
        })
    return pd.DataFrame(rows)


def best_sheet(runs: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = runs[runs.target == target]
        best = subset.sort_values("rmse").groupby("Experimento").first().reset_index()
        for _, row in best.sort_values("rmse").iterrows():
            rows.append({
                "Objetivo": target, "Experimento": row["Experimento"],
                "Mejor brazo": row["encoder"],
                "Encoder (checkpoint)": row["Encoder"],
                "Modelo de prediccion": row["Predictor"],
                "RMSE (test)": round(row["rmse"], 3),
                "MAE (test)": round(row["mae"], 3),
                "R2 (test)": round(row["r2"], 3),
                "Mejora RMSE sobre la media": round(baseline[target]["rmse"] - row["rmse"], 3),
            })
        rows.append({
            "Objetivo": target, "Experimento": "(referencia) predecir la media de entrenamiento",
            "Mejor brazo": "-", "Encoder (checkpoint)": "-",
            "Modelo de prediccion": "Dummy (media de entrenamiento)",
            "RMSE (test)": round(baseline[target]["rmse"], 3),
            "MAE (test)": round(baseline[target]["mae"], 3),
            "R2 (test)": round(baseline[target]["r2"], 3),
            "Mejora RMSE sobre la media": 0.0,
        })
    return pd.DataFrame(rows)


def _paired(runs: pd.DataFrame, contrast: str, question: str,
             left_mask, right_mask, left_name: str, right_name: str,
             key: str = "encoder") -> list[dict]:
    """One row per arm and target, the same arm measured both ways.

    Uniform column names across every contrast, with the condition spelled
    out in its own column. Naming the columns after each contrast instead
    would give a sheet mostly made of blanks.
    """
    left_rows = runs[left_mask(runs)].set_index([key, "target"])
    right_rows = runs[right_mask(runs)].set_index([key, "target"])
    left, right = left_rows["rmse"], right_rows["rmse"]
    checkpoints = left_rows["Encoder"]
    shared = left.index.intersection(right.index)
    rows = []
    for arm, target in sorted(shared):
        a, b = float(left.loc[(arm, target)]), float(right.loc[(arm, target)])
        rows.append({
            "Contraste": contrast, "Pregunta": question,
            "Brazo": arm, "Encoder (checkpoint)": checkpoints.loc[(arm, target)],
            "Objetivo": target,
            "Condicion A": left_name, "RMSE A (test)": round(a, 3),
            "Condicion B": right_name, "RMSE B (test)": round(b, 3),
            "Diferencia (B - A)": round(b - a, 3),
            "Gana": left_name if a < b else right_name,
        })
    return rows


# A checkpoint appears under different arm names depending on how it was
# used, so the paired contrasts need one key per checkpoint rather than per
# arm name.
CHECKPOINT_KEY = {
    "E0_TFIDF": "TF-IDF", "E0_TFIDF_QUESTION": "TF-IDF",
    "E1_BETO": "BETO", "E1_BETO_QUESTION": "BETO", "E5_BETO_FT": "BETO",
    "E2_E5SMALL": "e5-small", "E2_E5SMALL_QUESTION": "e5-small",
    "E5_E5SMALL_FT": "e5-small",
    "E2_E5BASE": "e5-base", "E2_E5BASE_QUESTION": "e5-base",
    "E5_E5BASE_FT": "e5-base",
    "E2_E5LARGE": "e5-large", "E2_E5LARGE_QUESTION": "e5-large",
    "E5_E5LARGE_FT": "e5-large",
    "E5_BERTIN_FT": "BERTIN", "E5_MDEBERTA_FT": "mDeBERTa-v3",
    "E5_XLMR_FT": "XLM-R",
}


def contrasts_sheet(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.assign(checkpoint=runs.encoder.map(CHECKPOINT_KEY).fillna(runs.encoder))
    notebook = runs.notebook.astype(str)
    rows = []
    rows += _paired(
        runs, "Aumentacion LLM", "Duplicar las filas de entrenamiento con parafrasis, ayuda?",
        lambda r: notebook.isin(["1", "2"]) & (r.augmentation == "no_augmentation"),
        lambda r: notebook.isin(["1", "2"]) & (r.augmentation == "augmented"),
        "sin aumentar", "aumentado")
    rows += _paired(
        runs, "Granularidad", "Entrenar por respuesta en vez de por participante, ayuda?",
        lambda r: (notebook == "1") & (r.augmentation == "no_augmentation"),
        lambda r: (notebook == "2") & (r.augmentation == "no_augmentation"),
        "nivel participante", "nivel pregunta", key="checkpoint")
    rows += _paired(
        runs, "Esquema de parada", "Validacion cruzada en vez de holdout unico, ayuda?",
        lambda r: r.family == "fine-tuning",
        lambda r: (r.family == "fine-tuning-cv") & (r.augmentation == "augmented"),
        "holdout unico", "validacion cruzada", key="checkpoint")
    rows += _paired(
        runs, "Fine-tuning vs congelado", "Dejar que el encoder aprenda la tarea, ayuda?",
        lambda r: (notebook == "2") & (r.augmentation == "augmented"),
        lambda r: (r.family == "fine-tuning-cv") & (r.augmentation == "augmented"),
        "congelado", "fine-tuned", key="checkpoint")
    # TSDAE compares one unadapted arm against four adapted checkpoints, so it
    # is not a pairing on a shared key and is built explicitly. The reference
    # is the unaugmented E2_E5SMALL, because notebook 3 is unaugmented too;
    # taking both augmentation conditions would put two rows under one target.
    base = runs[(notebook == "1") & (runs.encoder == "E2_E5SMALL")
                 & (runs.augmentation == "no_augmentation")].set_index("target")["rmse"]
    for _, row in runs[notebook == "3"].iterrows():
        reference = float(base.loc[row["target"]])
        rows.append({
            "Contraste": "Adaptacion TSDAE",
            "Pregunta": "Adaptar el encoder al vocabulario del corpus, ayuda?",
            "Brazo": row["encoder"], "Encoder (checkpoint)": row["Encoder"],
            "Objetivo": row["target"],
            "Condicion A": "e5-small sin adaptar", "RMSE A (test)": round(reference, 3),
            "Condicion B": "e5-small adaptado", "RMSE B (test)": round(float(row["rmse"]), 3),
            "Diferencia (B - A)": round(float(row["rmse"]) - reference, 3),
            "Gana": "e5-small sin adaptar" if reference < row["rmse"] else "e5-small adaptado",
        })
    return pd.DataFrame(rows).sort_values(["Contraste", "Objetivo", "Brazo"]).reset_index(drop=True)


def all_runs_sheet(runs: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "Experimento": "Experimento", "notebook": "Notebook", "encoder": "Brazo",
        "Encoder": "Encoder (checkpoint)", "Predictor": "Modelo de prediccion",
        "target": "Objetivo", "seed": "Semilla",
        "rmse": "RMSE (test)", "mae": "MAE (test)", "r2": "R2 (test)",
        "band_f1_macro": "F1 macro bandas", "band_qwk": "QWK bandas",
        "screening_f1": "F1 tamizaje", "screening_auc": "AUC tamizaje",
    }
    out = runs[list(columns)].rename(columns=columns)
    for column in ("RMSE (test)", "MAE (test)", "R2 (test)", "F1 macro bandas",
                    "QWK bandas", "F1 tamizaje", "AUC tamizaje"):
        out[column] = pd.to_numeric(out[column], errors="coerce").round(4)
    return out.sort_values(["Objetivo", "RMSE (test)"]).reset_index(drop=True)


def _autosize(worksheet, frame: pd.DataFrame, wide: set = ()) -> None:
    worksheet.freeze_panes = "A2"
    for position, column in enumerate(frame.columns, start=1):
        longest = max([len(str(column))] + [len(str(v)) for v in frame[column].head(200)])
        width = min(longest + 2, 95 if column in wide else 26)
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=position).column_letter].width = width


def main() -> int:
    runs = load()
    baseline = baseline_row()

    print(f"{len(runs)} runs, {runs.encoder.nunique()} distinct arms, "
          f"{runs.seed.nunique()} seed(s)")

    notes = pd.DataFrame([
        {"Apartado": "Que contiene",
         "Detalle": "Resultados del canal de texto: prediccion de PHQ-9 y GAD-7 a partir "
                     "de transcripciones de entrevista en espanol."},
        {"Apartado": "Particion",
         "Detalle": "158 participantes, divididos una sola vez en 110 de entrenamiento y "
                     "48 de test, estratificados por bandas de severidad PHQ-9 x GAD-7 con "
                     "semilla 42. Todos los brazos se evaluan sobre los mismos 48."},
        {"Apartado": "Modelo de prediccion",
         "Detalle": "En los experimentos congelados es el regresor clasico que gano la "
                     "validacion cruzada sobre los 110 de entrenamiento (SVR, RandomForest, "
                     "AdaBoost, ExtraTrees o GradientBoosting). En los de fine-tuning es el "
                     "propio encoder con una cabeza de regresion, entrenado de extremo a "
                     "extremo; no hay regresor aparte que elegir."},
        {"Apartado": "Metricas",
         "Detalle": "RMSE y MAE en puntos de la escala (menor es mejor). R2 es la fraccion "
                     "de varianza explicada: 0 equivale a predecir la media de "
                     "entrenamiento, negativo es peor que eso."},
        {"Apartado": "Referencia",
         "Detalle": f"Predecir la media de entrenamiento da RMSE "
                     f"{baseline['PHQ9']['rmse']:.3f} en PHQ-9 y "
                     f"{baseline['GAD7']['rmse']:.3f} en GAD-7. Un brazo que no supere "
                     f"eso no ha aprendido nada del texto."},
        {"Apartado": "ADVERTENCIA",
         "Detalle": "Cada brazo es una sola corrida con semilla 42, por lo que ninguna "
                     "cifra trae dispersion. Corridas previas con cinco semillas midieron "
                     "desviaciones de hasta 0.41 RMSE en los brazos afinados y tan bajas "
                     "como 0.003 en los congelados. Diferencias menores a ese orden no se "
                     "pueden distinguir del ruido del procedimiento."},
        {"Apartado": "Hoja 1",
         "Detalle": "Que se hizo en cada experimento y que buscaba medir."},
        {"Apartado": "Hoja 2",
         "Detalle": "El mejor brazo de cada experimento, por objetivo, con RMSE, MAE y R2."},
        {"Apartado": "Hoja 3",
         "Detalle": "Los contrastes declarados antes de correr: aumentacion, granularidad, "
                     "esquema de parada, fine-tuning frente a congelado y adaptacion TSDAE. "
                     "Cada fila compara el mismo checkpoint medido de las dos formas."},
        {"Apartado": "Hoja 4",
         "Detalle": f"Las {len(runs)} corridas registradas, con las metricas clinicas "
                     f"derivadas de la misma prediccion continua."},
    ])

    sheets = [
        ("Indice", notes, {"Detalle"}),
        ("1. Experimentos", experiments_sheet(runs), {"Que se hizo", "Que mide"}),
        ("2. Mejor por experimento", best_sheet(runs, baseline),
         {"Experimento", "Encoder (checkpoint)", "Modelo de prediccion"}),
        ("3. Contrastes", contrasts_sheet(runs), {"Pregunta", "Encoder (checkpoint)"}),
        ("4. Todas las corridas", all_runs_sheet(runs),
         {"Experimento", "Encoder (checkpoint)", "Modelo de prediccion"}),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUTPUT.name}")
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        for name, frame, wide in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
            _autosize(writer.sheets[name], frame, wide)
            print(f"  {name:28s} {frame.shape}")

    print(f"{OUTPUT}  ({OUTPUT.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
