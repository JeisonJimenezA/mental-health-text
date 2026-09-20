"""Task-adaptive pretraining (TAPT) with masked language modeling.

Continues a checkpoint's own pretraining objective on this study's unlabeled
text, so the encoder adapts to the register of the interviews (transcribed
speech of Colombian students describing their mood) without ever seeing a
label. Unlike TSDAE, which was run on the sentence-transformer, this keeps
the objective these models were built with: their MLM head is pretrained, and
there is no contrastive embedding geometry for the new objective to undo.

Two backbones are declared (config.MLM_BACKBONES), and each writes to its own
directory:

- BETO, whose frozen and fine-tuned arms are already measured, so the
  adaptation is the only thing that changes;
- ELiRF/RoBERTa-es-mental-large, which already carries a domain-adaptive
  pretraining of its own on 1.9M mental-health posts. Adapting it here is the
  DAPT-then-TAPT combination: someone else's domain, our register.

Data decisions, each of which affects what a later comparison means:

- Training participants only, never the held-out test set, and never the
  LLM paraphrases (the same reasons as the TSDAE corpus,
  src/domain_adaptation.py). The MentalRiskES patient messages can be added
  as a second, declared corpus variant.
- Whole documents, not sentences. MLM predicts a hidden word from its
  context, and a whole answer is more context than any sentence of it; it is
  also the unit the frozen encoder later embeds. An answer longer than the
  context is split at sentence boundaries, never mid-sentence, and a
  MentalRiskES example packs consecutive turns of one session. Text from two
  people is never joined into one example.
- A held-out share of the training participants (and one MentalRiskES
  session) is kept out of training to monitor MLM loss. Unlike TSDAE's
  reconstruction loss, MLM loss measures exactly what is being trained, so it
  is a meaningful signal of overfitting and forgetting. The interview
  participants held out are the same for both corpus variants.
- Anonymization placeholders ([LUGAR], [NOMBRE], ...) are unified and never
  masked: they are not language, and predicting them would teach nothing.

Masking is whole-word, as BETO was pretrained, and drawn afresh for every
batch, so every epoch sees different masks. Optionally, words of the clinical
lexicon (src/lexicon.py) are masked more often, keeping the overall rate.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import spacy
import torch
import transformers
from datasets import Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

from src import config
from src.lexicon import fold, is_clinical

SOURCE_INTERVIEWS = config.CORPUS_INTERVIEWS
SOURCE_MENTALRISKES = config.CORPUS_MENTALRISKES

_PLACEHOLDER_RE = re.compile(r"\[([^\[\]]{1,30})\]")
_nlp = None


def _spacy():
    """Lemmas for the lexicon and sentence boundaries for splitting long
    documents. The parser is disabled for speed, so boundaries come from the
    punctuation-based sentencizer instead."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        _nlp.add_pipe("sentencizer")
    return _nlp


def normalize_text(text: str) -> str:
    """Collapses whitespace and unifies placeholder spelling, so
    [INSTITUCIÓN] and [INSTITUCION] are the same string. Case, accents and
    punctuation are kept: BETO is cased and was trained on natural text."""
    text = " ".join(text.split())
    return _PLACEHOLDER_RE.sub(lambda m: f"[{fold(m.group(1)).upper()}]", text)


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _split_to_fit(text: str, tokenizer, limit: int) -> list[str]:
    """The text as one piece if it fits, otherwise consecutive pieces cut at
    sentence boundaries. A single sentence longer than the limit, which only
    an unpunctuated run-on produces, is cut between words."""
    if _n_tokens(tokenizer, text) <= limit:
        return [text]
    units = []
    for sentence in _spacy()(text).sents:
        if _n_tokens(tokenizer, sentence.text) <= limit:
            units.append(sentence.text)
            continue
        piece = []
        for word in sentence.text.split():
            if piece and _n_tokens(tokenizer, " ".join(piece + [word])) > limit:
                units.append(" ".join(piece))
                piece = []
            piece.append(word)
        if piece:
            units.append(" ".join(piece))
    return _pack(units, tokenizer, limit)


def _pack(units: list[str], tokenizer, limit: int) -> list[str]:
    """Joins consecutive units while the result still fits."""
    pieces, current = [], []
    for unit in units:
        if current and _n_tokens(tokenizer, " ".join(current + [unit])) > limit:
            pieces.append(" ".join(current))
            current = []
        current.append(unit)
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_documents(df, train_ids, cfg: config.MLMConfig, tokenizer) -> list[dict]:
    """[{source, group, text}], one per training example. `group` is the
    person the text belongs to (participant or MentalRiskES session)."""
    limit = cfg.max_tokens - 2   # [CLS] and [SEP]
    documents = []
    if SOURCE_INTERVIEWS in cfg.corpora:
        for participant in train_ids:
            for question in config.QUESTIONS:
                text = df.loc[participant, question]
                if not isinstance(text, str):
                    continue
                for piece in _split_to_fit(normalize_text(text), tokenizer, limit):
                    documents.append({"source": SOURCE_INTERVIEWS, "group": participant, "text": piece})
    if SOURCE_MENTALRISKES in cfg.corpora:
        from src.external_corpora import load_mentalriskes_sessions
        for session, turns in load_mentalriskes_sessions().items():
            units = [p for turn in turns for p in _split_to_fit(normalize_text(turn), tokenizer, limit)]
            for piece in _pack(units, tokenizer, limit):
                documents.append({"source": SOURCE_MENTALRISKES, "group": session, "text": piece})
    return documents


def split_validation(documents: list[dict], cfg: config.MLMConfig) -> tuple[list, list, dict]:
    """Holds out whole people. Each source draws from its own generator, so
    the interview participants held out do not depend on whether MentalRiskES
    is part of the corpus."""
    held_out = {}
    for source, n_or_fraction in [(SOURCE_INTERVIEWS, cfg.val_fraction),
                                  (SOURCE_MENTALRISKES, cfg.val_mentalriskes_sessions)]:
        groups = sorted({d["group"] for d in documents if d["source"] == source})
        if not groups:
            continue
        n_val = (max(1, round(len(groups) * n_or_fraction)) if isinstance(n_or_fraction, float)
                 else n_or_fraction)
        held_out[source] = sorted(random.Random(cfg.seed).sample(groups, n_val))
    val_groups = {g for groups in held_out.values() for g in groups}
    train = [d for d in documents if d["group"] not in val_groups]
    val = [d for d in documents if d["group"] in val_groups]
    return train, val, held_out


def encode_documents(documents: list[dict], tokenizer, cfg: config.MLMConfig) -> Dataset:
    """Token ids plus, per token, the word it belongs to (-1 when it must not
    be masked: special tokens and placeholders) and whether that word is in
    the clinical lexicon. Words are the tokenizer's own pre-tokenized words,
    which is what whole-word masking operates on."""
    records = {"input_ids": [], "word_ids": [], "lexicon": [], "length": [], "source": []}
    texts = [d["text"] for d in documents]
    for document, doc in zip(documents, _spacy().pipe(texts, batch_size=32)):
        text = document["text"]
        encoded = tokenizer(text, truncation=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) > cfg.max_tokens:
            raise AssertionError(f"Document exceeds {cfg.max_tokens} tokens after splitting")

        placeholders = [m.span() for m in _PLACEHOLDER_RE.finditer(text)]
        clinical = [(t.idx, t.idx + len(t.text)) for t in doc if is_clinical(t.text, t.lemma_)]

        def overlaps(spans, start, end):
            return any(s < end and start < e for s, e in spans)

        word_ids, lexicon = [], []
        for word_id, (start, end) in zip(encoded.word_ids(), encoded["offset_mapping"]):
            maskable = word_id is not None and end > start and not overlaps(placeholders, start, end)
            word_ids.append(word_id if maskable else -1)
            lexicon.append(int(maskable and overlaps(clinical, start, end)))
        records["input_ids"].append(encoded["input_ids"])
        records["word_ids"].append(word_ids)
        records["lexicon"].append(lexicon)
        records["length"].append(len(encoded["input_ids"]))
        records["source"].append(document["source"])
    return Dataset.from_dict(records)


class WholeWordMaskCollator:
    """Whole-word masking drawn at batch time, with optional lexicon weighting.

    A word is chosen with probability p; every subword piece of a chosen word
    becomes a prediction target and is replaced by [MASK] 80 percent of the
    time, a random token 10 percent, and left unchanged 10 percent, as in
    BERT. With lexicon masking, clinical words are chosen with p_lexicon and
    all others with the rate that keeps the expected share of chosen words at
    p within each document.

    Examples that already carry labels (the pre-masked validation sets) are
    only padded, so validation loss is measured on the same masks every epoch.
    """

    def __init__(self, tokenizer, cfg: config.MLMConfig, seed: int):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(seed)

    def word_probabilities(self, is_lexicon: torch.Tensor) -> torch.Tensor:
        p = self.cfg.mask_probability
        if not self.cfg.lexicon_masking:
            return torch.full(is_lexicon.shape, p)
        n, n_lexicon = len(is_lexicon), int(is_lexicon.sum())
        p_lexicon = self.cfg.lexicon_mask_probability
        p_other = (p * n - p_lexicon * n_lexicon) / (n - n_lexicon) if n > n_lexicon else 0.0
        p_other = min(max(p_other, 0.0), 1.0)
        return torch.where(is_lexicon, torch.tensor(p_lexicon), torch.tensor(p_other))

    def mask(self, example: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(example["input_ids"])
        word_ids = torch.tensor(example["word_ids"])
        token_lexicon = torch.tensor(example["lexicon"], dtype=torch.bool)

        words = torch.unique(word_ids[word_ids >= 0])
        labels = torch.full_like(input_ids, -100)
        if len(words) == 0:
            return input_ids, labels
        is_lexicon = torch.stack([token_lexicon[word_ids == w].any() for w in words])
        chosen = torch.rand(len(words), generator=self.generator) < self.word_probabilities(is_lexicon)
        if not chosen.any():
            chosen[torch.randint(len(words), (1,), generator=self.generator)] = True

        targets = torch.isin(word_ids, words[chosen])
        labels[targets] = input_ids[targets]
        draw = torch.rand(len(input_ids), generator=self.generator)
        input_ids = input_ids.clone()
        input_ids[targets & (draw < 0.8)] = self.tokenizer.mask_token_id
        random_positions = targets & (draw >= 0.8) & (draw < 0.9)
        input_ids[random_positions] = torch.randint(
            len(self.tokenizer), (int(random_positions.sum()),), generator=self.generator)
        return input_ids, labels

    def __call__(self, features: list[dict]) -> dict:
        pairs = []
        for feature in features:
            if feature.get("labels") is not None:
                pairs.append((torch.tensor(feature["input_ids"]), torch.tensor(feature["labels"])))
            else:
                pairs.append(self.mask(feature))
        width = max(len(ids) for ids, _ in pairs)
        batch_ids = torch.full((len(pairs), width), self.tokenizer.pad_token_id)
        batch_labels = torch.full((len(pairs), width), -100)
        attention = torch.zeros((len(pairs), width), dtype=torch.long)
        for i, (ids, labels) in enumerate(pairs):
            batch_ids[i, :len(ids)] = ids
            batch_labels[i, :len(labels)] = labels
            attention[i, :len(ids)] = 1
        return {"input_ids": batch_ids, "attention_mask": attention, "labels": batch_labels}


def premask(dataset: Dataset, collator: WholeWordMaskCollator) -> Dataset:
    """Fixes the masks of a validation set once, so its loss is comparable
    across epochs instead of moving with each new draw."""
    ids, labels = zip(*(collator.mask(example) for example in dataset))
    return Dataset.from_dict({"input_ids": [i.tolist() for i in ids],
                              "labels": [l.tolist() for l in labels]})


def masking_stats(dataset: Dataset, collator: WholeWordMaskCollator, passes: int = 3) -> dict:
    """Realized masking rates over the training set, as a check that the
    collator does what the configuration says."""
    maskable = lexicon = targets = lexicon_targets = 0
    for _ in range(passes):
        for example in dataset:
            _, labels = collator.mask(example)
            is_target = labels != -100
            word_ids = torch.tensor(example["word_ids"])
            is_lexicon = torch.tensor(example["lexicon"], dtype=torch.bool)
            maskable += int((word_ids >= 0).sum())
            lexicon += int(is_lexicon.sum())
            targets += int(is_target.sum())
            lexicon_targets += int((is_target & is_lexicon).sum())
    return {
        "target_share_of_maskable_tokens": targets / maskable,
        "lexicon_share_of_maskable_tokens": lexicon / maskable,
        "target_rate_lexicon_tokens": lexicon_targets / lexicon if lexicon else float("nan"),
        "target_rate_other_tokens": (targets - lexicon_targets) / (maskable - lexicon),
    }


def masked_lm_loss(outputs, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    """Cross-entropy over the masked tokens, normalized by every target in
    the gradient-accumulation window rather than per micro-batch.

    Passed to the Trainer explicitly because of a mismatch in transformers
    5.15: BertForMaskedLM.forward takes **kwargs, so the Trainer assumes the
    model normalizes by num_items_in_batch itself and skips dividing by the
    accumulation steps, but the model takes a plain per-batch mean. The loss,
    and with it every gradient, came out multiplied by grad_accum_steps.
    Summing and dividing by the window's target count fixes the scale and
    weights each masked token equally, whichever micro-batch it fell in.
    """
    logits = outputs.logits.float()
    labels = labels.to(logits.device)
    total = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum")
    if num_items_in_batch is None:   # evaluation
        return total / (labels != -100).sum().clamp(min=1)
    return total / num_items_in_batch


class _SaveDeclaredEpochs(TrainerCallback):
    def __init__(self, output_dir: Path, epochs: tuple[int, ...], tokenizer):
        self.output_dir, self.epochs, self.tokenizer = output_dir, set(epochs), tokenizer
        self.saved = {}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch in self.epochs and epoch not in self.saved:
            destination = self.output_dir / f"ep{epoch:03d}"
            if destination.exists():
                shutil.rmtree(destination)
            model.save_pretrained(destination)
            self.tokenizer.save_pretrained(destination)
            self.saved[epoch] = destination


def train_mlm(train_set: Dataset, val_sets: dict[str, Dataset], tokenizer, cfg: config.MLMConfig,
              output_dir: Path, device: str) -> tuple[dict, list[dict]]:
    """Trains for cfg.epochs, saving each declared epoch.
    Returns ({epoch: checkpoint path}, per-epoch loss history)."""
    transformers.set_seed(cfg.seed)
    # Both dropout names are BERT's and RoBERTa's alike; passing them as
    # config overrides keeps the checkpoint's own value when cfg.dropout is
    # None, which is what the BETO runs used.
    dropout = ({} if cfg.dropout is None
               else {"hidden_dropout_prob": cfg.dropout,
                     "attention_probs_dropout_prob": cfg.dropout})
    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model, **dropout)
    collator = WholeWordMaskCollator(tokenizer, cfg, seed=cfg.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir = output_dir / "_trainer"
    args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.use_fp16 and device == "cuda",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        remove_unused_columns=False,   # the collator needs word_ids and lexicon
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to="none",
    )
    saver = _SaveDeclaredEpochs(output_dir, cfg.checkpoint_epochs, tokenizer)
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_sets,
                      data_collator=collator, callbacks=[saver], compute_loss_func=masked_lm_loss)

    history = [{"epoch": 0, **_val_losses(trainer.evaluate())}]   # BETO before adaptation
    trainer.train()
    by_epoch = {}
    for entry in trainer.state.log_history:
        epoch = round(entry.get("epoch", 0))
        if "loss" in entry:
            by_epoch.setdefault(epoch, {})["train_loss"] = entry["loss"]
        if any(k.startswith("eval_") and k.endswith("_loss") for k in entry):
            by_epoch.setdefault(epoch, {}).update(_val_losses(entry))
    history += [{"epoch": epoch, **values} for epoch, values in sorted(by_epoch.items()) if epoch > 0]

    shutil.rmtree(trainer_dir, ignore_errors=True)
    return dict(sorted(saver.saved.items())), history


def _val_losses(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if key.startswith("eval_") and key.endswith("_loss"):
            source = key[len("eval_"):-len("_loss")]
            out[f"val_{source}_loss"] = value
            out[f"val_{source}_perplexity"] = float(np.exp(value))
    return out


def prepare(df, train_ids, test_ids, cfg: config.MLMConfig) -> dict:
    """Everything up to training: documents, validation split, encoded and
    pre-masked sets, and a realized-masking check. Shared by adapt() and the
    script's dry run."""
    leaked = set(train_ids) & set(test_ids)
    if leaked:
        raise AssertionError(f"Adaptation corpus would include test participants: {leaked}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if not tokenizer.is_fast:
        raise ValueError(f"{cfg.base_model} has no fast tokenizer; whole-word masking needs "
                         "word_ids() and character offsets to know word boundaries")
    documents = build_documents(df, train_ids, cfg, tokenizer)
    train_docs, val_docs, held_out = split_validation(documents, cfg)
    if not train_docs:
        raise ValueError("MLM training corpus is empty")

    train_set = encode_documents(train_docs, tokenizer, cfg)
    val_collator = WholeWordMaskCollator(tokenizer, cfg, seed=cfg.seed + 1)
    val_sets = {}
    for source in cfg.corpora:
        source_docs = [d for d in val_docs if d["source"] == source]
        if source_docs:
            val_sets[source] = premask(encode_documents(source_docs, tokenizer, cfg), val_collator)

    def describe(docs, dataset):
        return {source: {"documents": sum(d["source"] == source for d in docs),
                         "people": len({d["group"] for d in docs if d["source"] == source}),
                         "tokens": int(sum(n for n, s in zip(dataset["length"], dataset["source"])
                                           if s == source))}
                for source in cfg.corpora}

    return {
        "tokenizer": tokenizer,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "held_out": held_out,
        "train_set": train_set,
        "val_sets": val_sets,
        "train_summary": describe(train_docs, train_set),
        "val_summary": describe(val_docs, encode_documents(val_docs, tokenizer, cfg)) if val_docs else {},
        "masking": masking_stats(train_set, WholeWordMaskCollator(tokenizer, cfg, seed=cfg.seed + 2)),
    }


def adapt(df, train_ids, test_ids, cfg: config.MLMConfig, device: str) -> dict:
    """Prepares, trains, and writes the checkpoints with a record of how they
    were produced. Returns {epoch: checkpoint path}."""
    prepared = prepare(df, train_ids, test_ids, cfg)
    output_dir = config.MODELS_DIR / cfg.output_name
    print(f"train: {prepared['train_summary']}", flush=True)
    print(f"validation: {prepared['val_summary']}", flush=True)
    print(f"masking: {prepared['masking']}", flush=True)
    if device == "cpu":
        print("WARNING: no CUDA device found; this will be far slower on CPU", flush=True)

    checkpoints, history = train_mlm(prepared["train_set"], prepared["val_sets"],
                                     prepared["tokenizer"], cfg, output_dir, device)

    (output_dir / "mlm_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "config": cfg.as_dict(),
        "libraries": {"transformers": transformers.__version__, "torch": torch.__version__},
        "train": prepared["train_summary"],
        "validation": prepared["val_summary"],
        "validation_people": prepared["held_out"],
        "masking": prepared["masking"],
        "n_training_participants": len(train_ids),
        "device": device,
        "adapted_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoints": {str(epoch): path.name for epoch, path in checkpoints.items()},
    }
    (output_dir / "adaptation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return checkpoints
