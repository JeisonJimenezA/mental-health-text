"""TSDAE domain adaptation of the sentence-transformer (Phase C, step C1).

Adapts the E2 checkpoint to this project's own interview transcriptions with
an unsupervised denoising objective: a sentence is corrupted by deleting a
fraction of its words, and the model is trained to reconstruct the original
from the sentence embedding alone. No PHQ-9/GAD-7 label is involved, so the
resulting encoder can be compared against E2 as a test of domain adaptation
rather than of supervision (PROTOCOL.md, Section 9).

Three corpus decisions, each of which affects what the comparison means:

- Training participants only. Test transcriptions never enter adaptation,
  even though the objective is unsupervised: letting the encoder see them
  beforehand is transductive exposure ahead of the confirmatory evaluation.
- Original transcriptions only, never the LLM paraphrases. Paraphrases carry
  the generator's register rather than the students', they exist only for
  minority PHQ-9 classes, and making the corpus depend on the augmentation
  setting would yield two different adapted encoders, entangling adaptation
  with augmentation instead of keeping them separable factors.
- Sentences, not whole answers. The denoising objective operates at sentence
  level, and one document per participant would leave ~110 training examples.

The interviews can optionally be joined by the patient messages of the
MentalRiskES corpus (src/external_corpora.py), as a separate declared
variant written to its own directory. External text involves none of the
project's participants, so the held-out guarantee above is unaffected.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import sentence_transformers
import spacy
import torch
import transformers
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.models import Normalize
from torch import nn

from src import config, splits

_splitter = None


def _sentence_splitter():
    """Rule-based sentence segmentation.

    A blank Spanish pipeline with only the sentencizer, rather than the
    lemmatizer pipeline in encoders.py: that one runs with the parser
    disabled, which is what spaCy normally derives sentence boundaries from.
    Whisper's transcriptions are punctuated, so punctuation-based splitting
    is sufficient here.
    """
    global _splitter
    if _splitter is None:
        nlp = spacy.blank("es")
        nlp.add_pipe("sentencizer")
        _splitter = nlp
    return _splitter


def _ensure_nltk_tokenizer() -> None:
    """The deletion noise tokenizes with NLTK, which needs punkt on disk."""
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def split_sentences(texts, cfg: config.TSDAEConfig) -> list[str]:
    """Sentences from a sequence of texts, whatever their source.

    Fragments below min_sentence_words are dropped as too short to carry a
    reconstructable meaning; unpunctuated run-ons above max_sentence_words
    are split into consecutive pieces rather than discarded. Every corpus goes
    through this one function, so the sources differ in their text and never
    in how it was cut.
    """
    splitter = _sentence_splitter()
    sentences = []
    for text in texts:
        for sentence in splitter(text).sents:
            words = sentence.text.split()
            if len(words) < cfg.min_sentence_words:
                continue
            for start in range(0, len(words), cfg.max_sentence_words):
                piece = words[start:start + cfg.max_sentence_words]
                if len(piece) >= cfg.min_sentence_words:
                    sentences.append(" ".join(piece))
    return sentences


def build_sentence_corpus(df, participant_ids, cfg: config.TSDAEConfig,
                           questions: list[str] = config.QUESTIONS) -> list[str]:
    """Sentences from the given participants' original interview transcriptions."""
    texts = (df.loc[participant, question]
             for participant in participant_ids for question in questions)
    return split_sentences((t for t in texts if isinstance(t, str)), cfg)


def build_adaptation_corpus(df, participant_ids, cfg: config.TSDAEConfig) -> dict[str, list[str]]:
    """{corpus name: sentences} for every corpus cfg.corpora names.

    Kept per source rather than concatenated so the metadata can record how
    much each one contributed; train_tsdae shuffles them together.
    """
    corpus = {}
    for name in cfg.corpora:
        if name == config.CORPUS_INTERVIEWS:
            corpus[name] = build_sentence_corpus(df, participant_ids, cfg)
        elif name == config.CORPUS_MENTALRISKES:
            from src.external_corpora import load_mentalriskes_texts
            corpus[name] = split_sentences(load_mentalriskes_texts(), cfg)
        else:
            raise ValueError(f"Unknown corpus {name!r}")
    return corpus


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _delete_words(text: str, del_ratio: float, rng: np.random.Generator) -> str:
    """TSDAE's deletion noise, the same algorithm as
    DenoisingAutoEncoderDataset.delete but drawing from a generator owned by
    this run instead of numpy's global state, which the trainer reseeds.
    """
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text
    keep = rng.random(n) > del_ratio
    if not keep.any():
        keep[rng.integers(n)] = True  # guarantee that at least one word remains
    return TreebankWordDetokenizer().detokenize(np.array(words)[keep])


def _denoising_dataset(sentences: list[str], cfg: config.TSDAEConfig) -> Dataset:
    """(noisy, text) pairs whose noise is drawn again every time a row is read.

    The corruption must be applied lazily. model.fit() in sentence-transformers
    >= 3 iterates the DataLoader once and stores the result, so a noise
    function attached to the old DenoisingAutoEncoderDataset ran once per
    sentence and every epoch saw the same corrupted copy. With ~2100 fixed
    pairs the task becomes memorization rather than denoising, which is what
    the reconstruction loss of the first run showed (falling to 0.47 at 60
    percent deletion). set_transform runs at batch time instead, so each epoch
    draws a fresh deletion.

    The prefix is added after deletion, so it is never deleted, and to the
    encoder side only: the decoder reconstructs the sentence, not the prefix.
    Column order matters, since the loss reads (damaged, original).
    """
    rng = np.random.default_rng(cfg.seed)

    def corrupt(batch: dict) -> dict:
        texts = batch["text"]
        noisy = [cfg.encoder_prefix + _delete_words(t, cfg.deletion_ratio, rng) for t in texts]
        return {"noisy": noisy, "text": texts}

    dataset = Dataset.from_dict({"text": sentences})
    dataset.set_transform(corrupt)
    return dataset


class _SkipNormalize(nn.Module):
    """Runs a SentenceTransformer's modules except its final Normalize.

    Installed as the loss's encoder so the decoder receives the unnormalized
    pooled vector. It holds the model's own modules, so training updates the
    model that is saved, and that model still ends in Normalize.
    """

    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.sentence_model = model

    def forward(self, features: dict) -> dict:
        for module in self.sentence_model:
            if not isinstance(module, Normalize):
                features = module(features)
        return features


_STEP_RE = re.compile(r"(\d+)$")


def _organize_checkpoints(raw_dir: Path, output_dir: Path, steps_per_epoch: int) -> dict:
    """Renames the trainer's step-numbered checkpoints to epoch-numbered ones.

    Saving happens on a step interval because that is what the training API
    takes, but epochs are the unit this experiment reasons in, so each
    checkpoint is moved to ep<NN>/ and the raw directory removed.
    """
    saved = {}
    if not raw_dir.exists():
        return saved
    for path in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        match = _STEP_RE.search(path.name)
        if not match:
            continue
        epoch = round(int(match.group(1)) / steps_per_epoch)
        if epoch <= 0:
            continue
        destination = output_dir / f"ep{epoch:02d}"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(path), str(destination))
        saved[epoch] = destination
    shutil.rmtree(raw_dir, ignore_errors=True)
    return saved


def train_tsdae(sentences: list[str], cfg: config.TSDAEConfig, output_dir: Path,
                 device: str) -> dict:
    """Runs TSDAE for cfg.epochs, saving every cfg.checkpoint_every_epochs.
    Returns {epoch: checkpoint path}.

    One continuous run rather than several shorter ones: each run builds a
    fresh optimizer, so training in segments would reset Adam's moment
    estimates and the checkpoints would stop being points along a single
    trajectory.

    Uses SentenceTransformerTrainer directly rather than model.fit(). Besides
    freezing the noise (see _denoising_dataset), fit() builds its optimizer
    from the SentenceTransformer's parameters only. The decoder's own layers,
    its cross-attention and the LM head's transform, belong to the loss, so
    they were never updated and stayed at their random initialization for the
    whole run. The trainer builds the optimizer from the loss, which covers
    the encoder (through the tied weights) and the decoder alike.
    """
    _ensure_nltk_tokenizer()
    _seed_everything(cfg.seed)

    model = SentenceTransformer(cfg.base_model, device=device)
    dataset = _denoising_dataset(sentences, cfg)

    # decoder_name_or_path is ignored (and warned about) when the encoder and
    # decoder are tied, since the decoder is then initialized from the encoder.
    loss_kwargs = {"tie_encoder_decoder": cfg.tie_encoder_decoder}
    if not cfg.tie_encoder_decoder:
        loss_kwargs["decoder_name_or_path"] = cfg.base_model
    loss = DenoisingAutoEncoderLoss(model, **loss_kwargs)
    if not cfg.normalize_during_training:
        loss.encoder = _SkipNormalize(model)

    # The loss builds its decoder with from_pretrained, which lands on CPU and
    # in eval mode. The trainer switches only the SentenceTransformer to train
    # mode, so without this the decoder would run without dropout.
    loss.to(device)
    loss.train()

    decoder_only = [p for n, p in loss.decoder.named_parameters()
                    if all(p is not q for q in model.parameters())]
    if not decoder_only:
        raise AssertionError("Decoder has no parameters of its own; the tying is not what TSDAE expects")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw_checkpoints"
    steps_per_epoch = len(sentences) // cfg.batch_size   # drop_last

    args = SentenceTransformerTrainingArguments(
        output_dir=str(raw_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        dataloader_drop_last=True,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.scheduler,
        warmup_steps=0,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        save_strategy="steps",
        save_steps=steps_per_epoch * cfg.checkpoint_every_epochs,
        save_total_limit=None,          # keep every checkpoint
        save_only_model=True,           # optimizer state is ~2x the model and never resumed
        logging_steps=steps_per_epoch,  # one loss reading per epoch
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)

    # Every decoder-only parameter must reach the optimizer; that they did not
    # is the defect this function exists to avoid.
    optimizer = trainer.create_optimizer()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = sum(id(p) not in optimized for p in decoder_only)
    if missing:
        raise AssertionError(f"{missing} decoder parameters are not being optimized")

    trainer.train()
    loss_history = [{"epoch": round(entry["epoch"], 2), "loss": entry["loss"]}
                    for entry in trainer.state.log_history if "loss" in entry]
    (output_dir / "loss_history.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")

    saved = _organize_checkpoints(raw_dir, output_dir, steps_per_epoch)

    # The in-memory model is the final epoch, saved explicitly rather than
    # relying on the trainer having written a checkpoint on the last step.
    final_dir = output_dir / f"ep{cfg.epochs:02d}"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    model.save(str(final_dir))
    saved[cfg.epochs] = final_dir
    return dict(sorted(saved.items()))


def _participant_embeddings(df, participant_ids, model_name: str, device: str) -> np.ndarray:
    from src import encoders
    model = encoders.load_sentence_transformer(model_name, device)
    return np.vstack([encoders.st_embed_participant(df.loc[pid], model, device)
                       for pid in participant_ids])


def intrinsic_report(df, participant_ids, model_name: str, device: str) -> dict:
    """Diagnostics for one encoder's space, computed on training participants
    only (PROTOCOL.md, Section 9).

    Answers a question the reconstruction loss cannot: whether the adapted
    encoder still produces a usable space. Three readings:

    - cosine spread: TSDAE's known failure mode is collapse, where every
      sentence maps to nearly the same vector. A mean off-diagonal cosine
      near 1 with almost no spread means the space collapsed, and any
      downstream result would say nothing about domain adaptation.
    - band silhouette: whether participants in the same PHQ-9 severity band
      sit closer together than participants in different bands.
    - score-distance correlation: whether embedding distance tracks the gap
      in PHQ-9 score, which is what the downstream regressor needs.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    X = _participant_embeddings(df, participant_ids, model_name, device)
    sims = cosine_similarity(X)
    off_diagonal = sims[~np.eye(len(sims), dtype=bool)]

    scores = df.loc[participant_ids, config.TARGET_COLUMNS[config.PRIMARY_TARGET]].to_numpy()
    bands = np.array([splits.severity_band(s) for s in scores])

    triu = np.triu_indices(len(X), k=1)
    embedding_distance = 1 - sims[triu]
    score_gap = np.abs(scores[:, None] - scores[None, :])[triu]
    rho, p_value = spearmanr(embedding_distance, score_gap)

    return {
        "model": model_name,
        "n_participants": len(participant_ids),
        "dim": int(X.shape[1]),
        "cosine_mean": float(off_diagonal.mean()),
        "cosine_std": float(off_diagonal.std()),
        "cosine_p05": float(np.percentile(off_diagonal, 5)),
        "cosine_p95": float(np.percentile(off_diagonal, 95)),
        "band_silhouette": float(silhouette_score(X, bands, metric="cosine")),
        "score_distance_spearman": float(rho),
        "score_distance_p": float(p_value),
    }


def nearest_neighbours(df, participant_ids, model_name: str, device: str,
                        n_examples: int = 3, k: int = 3) -> list[dict]:
    """A few participants and their closest neighbours, for a qualitative read
    on whether the space still groups similar interviews together.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    X = _participant_embeddings(df, participant_ids, model_name, device)
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, -np.inf)

    target_col = config.TARGET_COLUMNS[config.PRIMARY_TARGET]
    examples = []
    for i in range(min(n_examples, len(participant_ids))):
        order = np.argsort(sims[i])[::-1][:k]
        examples.append({
            "participant": participant_ids[i],
            "phq9": float(df.loc[participant_ids[i], target_col]),
            "neighbours": [
                {"participant": participant_ids[j],
                 "phq9": float(df.loc[participant_ids[j], target_col]),
                 "cosine": float(sims[i, j])}
                for j in order
            ],
        })
    return examples


def adapt(df, train_ids, test_ids, cfg: config.TSDAEConfig, device: str) -> dict:
    """Builds the corpus, checks it against the held-out participants, trains,
    and writes the checkpoints plus a record of how they were produced.
    Returns {epoch: checkpoint path}.
    """
    leaked = set(train_ids) & set(test_ids)
    if leaked:
        raise AssertionError(f"Adaptation corpus would include test participants: {leaked}")

    corpus = build_adaptation_corpus(df, train_ids, cfg)
    for name, source_sentences in corpus.items():
        if not source_sentences:
            raise ValueError(f"Adaptation corpus {name!r} is empty")
    sentences = [s for source_sentences in corpus.values() for s in source_sentences]

    output_dir = config.MODELS_DIR / cfg.output_name
    print(f"corpus: {len(sentences)} sentences "
          f"({', '.join(f'{name}: {len(s)}' for name, s in corpus.items())}); "
          f"interviews from {len(train_ids)} training participants", flush=True)
    print(f"training TSDAE on {device}, {cfg.epochs} epochs, batch {cfg.batch_size}", flush=True)
    if device == "cpu":
        print("WARNING: no CUDA device found; this will be far slower on CPU", flush=True)

    checkpoints = train_tsdae(sentences, cfg, output_dir, device)

    metadata = {
        "config": cfg.as_dict(),
        # The first run's defects came from how one library version implemented
        # fit(), so the versions are part of what produced these checkpoints.
        "libraries": {"sentence_transformers": sentence_transformers.__version__,
                      "transformers": transformers.__version__,
                      "torch": torch.__version__},
        "n_sentences": len(sentences),
        "n_sentences_by_corpus": {name: len(s) for name, s in corpus.items()},
        "n_participants": len(train_ids),
        "device": device,
        "adapted_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoints": {str(epoch): path.name for epoch, path in checkpoints.items()},
    }
    (output_dir / "adaptation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return checkpoints
