"""Text encoders: participant transcriptions to feature matrices.

Three families, one per experimental arm:

- E0, TF-IDF: a bag-of-words vectorizer. This is the only encoder whose
  input is normalized first (numbers replaced, then lemmatized), because it
  has no way to relate inflected forms of one word and its vocabulary
  budget is small. Vocabulary fitting happens inside each cross-validation
  fold, via the sklearn Pipeline, rather than once beforehand.
- E1, BETO: frozen, attention-masked mean pooling over the last hidden
  state.
- E2 (and E3 in Phase C), sentence-transformer of the e5 family: frozen,
  pooled by the model's own forward pass, with e5's required "query: "
  prefix on every chunk.

BETO and e5 take the raw transcription: they are subword-tokenized models
pretrained on natural, cased Spanish text and learn casing and inflection
from context, so normalizing their input would push it outside the
pretraining distribution.

Both dense encoders share the same sliding-window chunking for
transcriptions longer than the model's context, and average the resulting
chunk vectors, then average across a participant's available questions.
Encoder behaviour is unchanged from the exploratory notebooks
(01_tfidf, 02_beto_embeddings, 021_sentence_transformers_ml).
"""
import re
import unicodedata

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from src import config

E5_PREFIX = config.E5_PREFIX


# --- Shared helpers -----------------------------------------------------

def chunk_starts(n_tokens: int, chunk_size: int, stride: int) -> list[int]:
    """Sliding-window start offsets covering n_tokens."""
    if n_tokens <= chunk_size:
        return [0]
    starts = []
    for start in range(0, n_tokens, stride):
        starts.append(start)
        if start + chunk_size >= n_tokens:
            break
    return starts


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# --- E0: TF-IDF ---------------------------------------------------------

_NUMBER_RE = re.compile(r"\d+")
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    return _nlp


def normalize_numbers(text: str) -> str:
    """Replace digit runs with a placeholder. A specific number mentioned in
    an interview (an age, a count of days) is high-cardinality and rarely
    repeats across participants, so it consumes vocabulary budget without
    generalizing.
    """
    return _NUMBER_RE.sub(" NUM ", text)


def lemmatize(text: str) -> str:
    """Collapse Spanish verb conjugation and gender/number agreement, which
    would otherwise split one concept across surface forms that each fall
    below min_df in this small corpus.
    """
    doc = _get_nlp()(text)
    return " ".join(tok.lemma_ for tok in doc if not tok.is_space)


def preprocess_for_tfidf(text: str, lemmatize_text: bool = True,
                          normalize_numbers_text: bool = True) -> str:
    if normalize_numbers_text:
        text = normalize_numbers(text)
    return lemmatize(text) if lemmatize_text else text


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def spanish_stopwords() -> list[str]:
    """Spanish stopwords plus unaccented variants.

    TfidfVectorizer strips accents from documents (strip_accents="unicode")
    but not from the stop_words list, so unaccented variants must be added
    explicitly or common words like "mas"/"tambien" leak into the vocabulary.
    """
    try:
        base = stopwords.words("spanish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = stopwords.words("spanish")
    words = set(base)
    return sorted(words | {_strip_accents(w) for w in words})


def participant_documents(df: pd.DataFrame, cfg: config.TrainingConfig | None = None,
                           questions: list[str] = config.QUESTIONS) -> pd.Series:
    """One concatenated document per participant from their available
    answers, normalized as cfg asks.
    """
    cfg = cfg or config.TrainingConfig()
    texts = df[questions].apply(lambda row: " ".join(t for t in row if isinstance(t, str)), axis=1)
    if cfg.tfidf_lemmatize or cfg.tfidf_normalize_numbers:
        texts = texts.apply(lambda t: preprocess_for_tfidf(
            t, lemmatize_text=cfg.tfidf_lemmatize, normalize_numbers_text=cfg.tfidf_normalize_numbers))
    return texts


# Distinguishes "the caller said nothing" from "the caller said no cap".
# TfidfVectorizer reads max_features=None as unlimited, so `or` would make
# that request silently fall through to the configured cap instead.
UNSET = object()


def build_vectorizer(cfg: config.TrainingConfig | None = None,
                      ngram_range: tuple[int, int] | None = None,
                      max_features=UNSET) -> TfidfVectorizer:
    """The E0 vectorizer, or a variant of it.

    The overrides exist so another arm can widen the n-gram window without
    changing what E0 does. E0's rows are already logged under the
    configuration's own settings, and sharing one vectorizer would silently
    redefine the baseline every other arm is measured against.

    `max_features=None` means no cap, and omitting it takes the configured
    one. `ngram_range` has no unlimited value, so None there simply defers to
    the configuration.
    """
    cfg = cfg or config.TrainingConfig()
    return TfidfVectorizer(
        stop_words=spanish_stopwords(), strip_accents="unicode", sublinear_tf=True,
        max_features=(cfg.tfidf_max_features if max_features is UNSET else max_features),
        ngram_range=tuple(ngram_range or cfg.tfidf_ngram_range),
        min_df=cfg.tfidf_min_df, max_df=cfg.tfidf_max_df,
    )


# --- E1: BETO, frozen ---------------------------------------------------

def load_beto(device: str, model_name: str = config.BETO_MODEL):
    """BETO, or a checkpoint adapted from it (e.g. models/mlm_beto/ep010).

    An MLM-adapted checkpoint is saved as BertForMaskedLM, which has no
    pooler; AutoModel then initializes one at random. That is harmless here
    because the embedding is a mean over the last hidden state, which never
    passes through the pooler.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def beto_embed_text(text: str, tokenizer, model, device: str,
                     max_length: int = config.MAX_TOKENS,
                     stride: int = config.CHUNK_STRIDE) -> np.ndarray:
    chunk_size = max_length - 2
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    def forward(ids: list[int]) -> np.ndarray:
        input_ids = torch.tensor([[tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]]).to(device)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask)
        m = mask.unsqueeze(-1).float()
        return ((out.last_hidden_state * m).sum(1) / m.sum(1))[0].cpu().numpy()

    chunks = [forward(token_ids[s:s + chunk_size]) for s in chunk_starts(len(token_ids), chunk_size, stride)]
    return np.mean(chunks, axis=0)


def beto_embed_participant(row, tokenizer, model, device: str,
                            questions: list[str] = config.QUESTIONS,
                            max_length: int = config.MAX_TOKENS,
                            stride: int = config.CHUNK_STRIDE) -> np.ndarray | None:
    vectors = [beto_embed_text(row[q], tokenizer, model, device, max_length, stride)
                for q in questions if isinstance(row[q], str)]
    return np.mean(vectors, axis=0) if vectors else None


# --- E2/E3: sentence-transformer (e5), frozen ---------------------------

def load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def st_embed_text(text: str, model: SentenceTransformer, device: str, prefix: str = E5_PREFIX,
                   max_length: int = config.MAX_TOKENS,
                   stride: int = config.CHUNK_STRIDE) -> np.ndarray:
    """The prefix goes on every chunk, not only the first, so each chunk sees
    the input distribution the model was trained on. Chunk outputs are
    unit-norm but their mean is not, so the average is renormalized: leaving
    it unnormalized would turn "how many chunks were averaged" into a
    spurious feature.
    """
    tokenizer = model.tokenizer
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    chunk_size = max_length - 2 - len(prefix_ids)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    def forward(ids: list[int]) -> np.ndarray:
        input_ids = torch.tensor([[tokenizer.cls_token_id] + prefix_ids + ids + [tokenizer.sep_token_id]]).to(device)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model.forward({"input_ids": input_ids, "attention_mask": mask})
        return out["sentence_embedding"][0].float().cpu().numpy()

    chunks = [forward(token_ids[s:s + chunk_size])
               for s in chunk_starts(len(token_ids), chunk_size, stride)]
    return chunks[0] if len(chunks) == 1 else l2_normalize(np.mean(chunks, axis=0))


def st_embed_participant(row, model: SentenceTransformer, device: str,
                          questions: list[str] = config.QUESTIONS,
                          prefix: str = E5_PREFIX,
                          max_length: int = config.MAX_TOKENS,
                          stride: int = config.CHUNK_STRIDE) -> np.ndarray | None:
    vectors = [st_embed_text(row[q], model, device, prefix, max_length, stride)
                for q in questions if isinstance(row[q], str)]
    return l2_normalize(np.mean(vectors, axis=0)) if vectors else None
