"""Classical regressors and their hyperparameter search space, shared across
frozen-encoder arms (E0, E1, E2, E3). The model set, search space and search
procedure (RandomizedSearchCV, n_iter=50, scoring=RMSE) are unchanged from the
exploratory phase (02_beto_embeddings.ipynb), so only the seed and the
preprocessing step (TF-IDF vectorizer vs. a scaler over precomputed
embeddings) vary across encoder arms.
"""
import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src import config

BASELINE_KEY = "Dummy(mean)"

# Penalties RidgeCV chooses between. Wide on purpose: the matrices here run
# from 25 columns against 110 rows to 385 against 376, and the alpha that
# suits one is orders of magnitude from the alpha that suits another.
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
N_ITER_SEARCH = config.N_SEARCH_ITER


def build_regressors(random_state: int, names: tuple[str, ...] = ()) -> dict:
    """Every candidate regressor, or the subset `names` asks for.

    Narrowing the set narrows what cross-validation can pick from, so a run
    under a restricted set is not comparable with one under the full set.
    That is why the subset lives in TrainingConfig and is logged with every
    row rather than being a local argument at the call site.

    The mean baseline is always kept: nothing ever selects it, and the code
    paths that fit it explicitly still want it present.
    """
    regressors = _all_regressors(random_state)
    if not names:
        return regressors
    unknown = set(names) - set(regressors)
    if unknown:
        raise ValueError(f"Unknown regressor(s) {sorted(unknown)}; "
                         f"expected from {sorted(set(regressors) - {BASELINE_KEY})}")
    return {name: regressors[name] for name in (*names, BASELINE_KEY)}


def _all_regressors(random_state: int) -> dict:
    return {
        # The linear model of the set. RidgeCV rather than plain Ridge,
        # because with n_search_iter=0 nothing tunes a penalty and Ridge's
        # default alpha=1.0 is far too weak where p approaches n and several
        # columns are collinear: it reached a cross-validated RMSE of 25.7 on
        # the 60-feature lexical matrix, against 5.94 for the mean predictor.
        # RidgeCV picks alpha over the grid below by leave-one-out, which it
        # solves in closed form, so it costs no nested search -- the same
        # reason the others are fitted at their defaults.
        "RidgeCV": RidgeCV(alphas=RIDGE_ALPHAS),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.5),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                               min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                                           min_samples_leaf=2, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=4, subsample=0.8,
                                                        random_state=random_state),
        "LightGBM": LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                    colsample_bytree=0.8, subsample=0.8,
                                    random_state=random_state, n_jobs=-1, verbose=-1),
        "KNN": KNeighborsRegressor(n_neighbors=7, metric="cosine"),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=3, random_state=random_state),
        "AdaBoost": AdaBoostRegressor(n_estimators=200, learning_rate=0.5, random_state=random_state),
        BASELINE_KEY: DummyRegressor(strategy="mean"),
    }


PARAM_GRIDS = {
    # Only reached when n_search_iter > 0. At the default the estimator
    # already searches RIDGE_ALPHAS internally, so this widens the range
    # rather than introducing a search where there was none.
    "RidgeCV": {"reg__alphas": [RIDGE_ALPHAS, np.logspace(-4, 6, 41)]},
    "SVR": {"reg__C": loguniform(1e-1, 1e2), "reg__epsilon": uniform(0.05, 1.95),
            "reg__gamma": ["scale", "auto"]},
    "RandomForest": {
        "reg__n_estimators": randint(100, 500), "reg__max_features": ["sqrt", "log2", 0.3],
        "reg__min_samples_leaf": randint(1, 6), "reg__max_depth": [None, 5, 10, 20],
    },
    "ExtraTrees": {
        "reg__n_estimators": randint(100, 500), "reg__max_features": ["sqrt", "log2", 0.3],
        "reg__min_samples_leaf": randint(1, 6), "reg__max_depth": [None, 5, 10, 20],
    },
    "GradientBoosting": {
        "reg__n_estimators": randint(100, 400), "reg__learning_rate": loguniform(1e-2, 3e-1),
        "reg__max_depth": randint(2, 6), "reg__subsample": uniform(0.6, 0.4),
    },
    "LightGBM": {
        "reg__n_estimators": randint(100, 400), "reg__num_leaves": randint(15, 64),
        "reg__learning_rate": loguniform(1e-2, 3e-1), "reg__colsample_bytree": uniform(0.6, 0.4),
        "reg__subsample": uniform(0.6, 0.4),
    },
    "KNN": {
        "reg__n_neighbors": randint(3, 16), "reg__metric": ["cosine", "euclidean", "manhattan"],
        "reg__weights": ["uniform", "distance"],
    },
    "DecisionTree": {
        "reg__max_depth": [3, 4, 5, 7, 10, None], "reg__min_samples_leaf": randint(1, 8),
        "reg__max_features": ["sqrt", "log2", None],
    },
    "AdaBoost": {"reg__n_estimators": randint(50, 300), "reg__learning_rate": loguniform(1e-2, 1.0)},
}


def make_pipeline(reg, preprocessor) -> Pipeline:
    """preprocessor is StandardScaler() for precomputed features, or a
    TfidfVectorizer / ColumnTransformer for raw text, so vocabulary fitting
    stays inside each CV fold.

    Two steps, not three: every candidate here reads the sparse matrix the
    vectorizer emits, so nothing has to be densified on the way through.
    That is a constraint on which regressors may join the set, and it is why
    BayesianRidge, ARDRegression and PLSRegression are not in it.
    """
    return Pipeline([("pre", clone(preprocessor)), ("reg", clone(reg))])


class DefaultFit:
    """A fitted default-parameter pipeline, exposing the same attributes as a
    fitted RandomizedSearchCV so callers do not branch on which path ran."""

    def __init__(self, estimator, cv_score: float):
        self.best_estimator_ = estimator
        self.best_score_ = cv_score
        self.best_params_ = {}


def search_best_model(X_train, y_train, preprocessor, cv, seed: int,
                       n_iter: int = N_ITER_SEARCH,
                       names: tuple[str, ...] = ()) -> dict:
    """Scores every candidate regressor over the given CV folds and returns
    the fitted result per model name.

    The mean predictor is scored too, so every arm's trial log carries the
    number its models have to beat, but best_of never selects it: an arm
    reports the best real model plus, in the log beside it, what doing
    nothing would have cost.

    n_iter=0 fits each candidate with its library defaults; cross-validation
    still selects the family, only the within-family tuning is skipped. Any
    n_iter above 0 runs a random search of that size instead.

    cv must be a materialized list of (train_idx, test_idx) tuples, not a
    splitter's .split() generator: the same cv is reused across every
    regressor's .fit() call in the loop below, and a generator would be
    exhausted after the first one.
    """
    regressors = build_regressors(seed, names)
    results = {}
    for name, reg in regressors.items():
        pipeline = make_pipeline(reg, preprocessor)
        if n_iter and n_iter > 0 and name != BASELINE_KEY:
            search = RandomizedSearchCV(
                pipeline, PARAM_GRIDS[name],
                n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                random_state=seed, n_jobs=-1, refit=True,
            )
            search.fit(X_train, y_train)
            results[name] = search
        else:
            # n_jobs=1 on purpose: the defaults path is only 5 fits per model,
            # and spawning a worker pool costs more than the parallelism saves
            # (measured 22s against 37s for the TF-IDF arm).
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                      scoring="neg_root_mean_squared_error", n_jobs=1)
            pipeline.fit(X_train, y_train)   # refit on all training data, as refit=True does
            results[name] = DefaultFit(pipeline, float(np.mean(scores)))
    return results


def tune_model(model_name: str, X_train, y_train, preprocessor, cv, seed: int,
                n_iter: int = config.SELECTIVE_TUNING_ITER):
    """Runs a random search for one already-chosen model family.

    Choose the family from its cross-validation ranking, not from test
    performance: picking what to tune by looking at the held-out set is
    selection on the test set by another name.
    """
    search = RandomizedSearchCV(
        make_pipeline(build_regressors(seed)[model_name], preprocessor),
        PARAM_GRIDS[model_name],
        n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
        random_state=seed, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search


def best_of(results: dict):
    """Model with the lowest CV RMSE (highest neg_root_mean_squared_error).

    The mean predictor is excluded. It is fitted and logged so the trial
    table says what the arm had to beat, but selecting it would mean
    reporting a constant as the study's answer.
    """
    candidates = [k for k in results if k != BASELINE_KEY] or list(results)
    name = max(candidates, key=lambda k: results[k].best_score_)
    return name, results[name].best_estimator_
