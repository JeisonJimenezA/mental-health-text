"""Vocabulary diagnostic: does the domain's vocabulary need enriching?

Before building a corpus to enrich an encoder's vocabulary, measures whether
the candidate tokenizers already represent this domain's words well. A word
the tokenizer keeps whole has its own learned embedding; a word cut into
several subword pieces has to be reassembled from fragments by the upper
layers, which is where a small model trained on general text struggles. If
the domain's content and clinical words are rarely split, extending the
vocabulary has little to add and the effort belongs in the corpus and the
training objective instead.

Three readings per tokenizer and corpus:
- fertility: subword pieces per word, and the share of words split at all,
  for every word and for content words (nouns, verbs, adjectives, adverbs);
- clinical lexicon: the same, restricted to the vocabulary of the PHQ-9 and
  GAD-7 items plus common colloquial terms for the same states;
- the most frequent content words each tokenizer splits, for a qualitative
  read of what is actually being fragmented.

Text only, no labels, no training. The interview corpus is restricted to the
training participants, like every other decision taken before evaluation.
Tokenizers are read from the local Hugging Face cache and skipped if absent;
nothing is downloaded.

Usage:
  ..\\env\\Scripts\\python.exe scripts\\vocabulary_diagnostic.py
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import spacy
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src import config, data, splits
from src.external_corpora import load_mentalriskes_texts
from src.lexicon import fold, is_clinical

hf_logging.set_verbosity_error()

OUTPUT_DIR = config.RESULTS_DIR / "diagnostics" / "vocabulary"

# Tokenizer families the study uses or could adopt. The e5 checkpoints share
# XLM-R's tokenizer, so one entry stands for all three sizes.
TOKENIZERS = {
    "BETO (cased)": config.BETO_MODEL,
    "e5 / XLM-R": config.SENTENCE_TRANSFORMER_MODEL,
    "BERTIN (RoBERTa-es)": "bertin-project/bertin-roberta-base-spanish",
    "RoBERTa biomedical-clinical-es": "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
    "mRoBERTa (BSC)": "BSC-LT/mRoBERTa",
    "mDeBERTa-v3": "microsoft/mdeberta-v3-base",
}

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def load_corpora() -> dict[str, list[str]]:
    df = data.load_dataset()
    train_ids, _ = splits.train_test_participants(df)
    interviews = [df.loc[p, q] for p in train_ids for q in config.QUESTIONS
                  if isinstance(df.loc[p, q], str)]
    return {"interviews (train)": interviews, "mentalriskes": load_mentalriskes_texts()}


def word_spans(texts: list[str]) -> list[list[dict]]:
    """Per text, its words with character span, content-word flag and
    lexicon flag.

    Placeholders such as [LUGAR] are not words of the language and are left
    out, as are numbers and punctuation.
    """
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    result = []
    for doc in nlp.pipe(texts, batch_size=64):
        words = []
        for i, token in enumerate(doc):
            if not token.is_alpha or (i > 0 and doc[i - 1].text == "["):
                continue
            words.append({
                "form": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text),
                "content": token.pos_ in CONTENT_POS,
                "lexicon": is_clinical(token.text, token.lemma_),
            })
        result.append(words)
    return result


def count_pieces(tokenizer, texts: list[str], spans: list[list[dict]]) -> list[dict]:
    """One record per word occurrence with the subword pieces covering it.

    Counted in context: the whole text is tokenized once and each piece is
    assigned to the word its character offsets overlap. Tokenizing words in
    isolation misstates the count, since how a word start is marked (a space,
    a separate "▁" piece) depends on the tokenizer and on what precedes it.
    Zero-width pieces are ignored.
    """
    records = []
    for text, words in zip(texts, spans):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        offsets = [(s, e, t) for (s, e), t in zip(encoded["offset_mapping"], tokens) if e > s]
        cursor = 0
        for word in words:
            while cursor < len(offsets) and offsets[cursor][1] <= word["start"]:
                cursor += 1
            covering = []
            j = cursor
            while j < len(offsets) and offsets[j][0] < word["end"]:
                covering.append(offsets[j][2])
                j += 1
            records.append({**word, "n_pieces": len(covering), "pieces": " | ".join(covering),
                            "unk": tokenizer.unk_token in covering})
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpora = load_corpora()
    spans = {name: word_spans(texts) for name, texts in corpora.items()}

    tokenizers = {}
    for label, name in TOKENIZERS.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception as exc:
            print(f"[skip] {label} ({name}): not in local cache ({type(exc).__name__})")
            continue
        # A cached config without vocabulary files still "loads" and then
        # tokenizes everything to nothing; offsets need a fast tokenizer.
        if not tokenizer.is_fast or not tokenizer.tokenize("hola"):
            print(f"[skip] {label} ({name}): no usable fast tokenizer in local cache")
            continue
        tokenizers[label] = tokenizer

    summary, occurrences = [], []
    for corpus, texts in corpora.items():
        for label, tokenizer in tokenizers.items():
            records = pd.DataFrame(count_pieces(tokenizer, texts, spans[corpus]))
            records["corpus"], records["tokenizer"] = corpus, label
            occurrences.append(records)

            row = {"corpus": corpus, "tokenizer": label, "vocab_size": len(tokenizer)}
            for group, subset in [("all", records), ("content", records[records.content]),
                                  ("lexicon", records[records.lexicon])]:
                row[f"{group}_fertility"] = subset.n_pieces.mean()
                row[f"{group}_pct_split"] = 100 * (subset.n_pieces > 1).mean()
            row["pct_unk"] = 100 * records.unk.mean()
            summary.append(row)

    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    occurrences = pd.concat(occurrences, ignore_index=True)
    fragmented = (occurrences[occurrences.content & (occurrences.n_pieces > 1)]
                  .groupby(["corpus", "tokenizer", "form", "pieces"], as_index=False)
                  .agg(occurrences=("n_pieces", "size"), lexicon=("lexicon", "any"))
                  .sort_values(["corpus", "tokenizer", "occurrences"], ascending=[True, True, False]))
    fragmented.to_csv(OUTPUT_DIR / "fragmented_content_words.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n=== fertility and share of words split (pct) ===")
    print(summary.drop(columns=["vocab_size"]).round(2).to_string(index=False))
    for corpus in corpora:
        lexicon_forms = occurrences[(occurrences.corpus == corpus) & occurrences.lexicon].form
        print(f"\n=== {corpus}: {lexicon_forms.map(fold).nunique()} distinct lexicon forms, "
              f"{len(lexicon_forms) // max(len(tokenizers), 1)} occurrences ===")
        for label in tokenizers:
            top = fragmented[(fragmented.corpus == corpus) & (fragmented.tokenizer == label)]
            lex = top[top.lexicon].head(10)
            print(f"\n[{label}] most frequent split content words:")
            print("  " + "; ".join(f"{r.form} ({r.occurrences}) -> {r.pieces}"
                                   for r in top.head(12).itertuples()))
            print(f"[{label}] most frequent split lexicon words:")
            print("  " + ("; ".join(f"{r.form} ({r.occurrences}) -> {r.pieces}"
                                    for r in lex.itertuples()) or "none"))
    print(f"\nsaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
