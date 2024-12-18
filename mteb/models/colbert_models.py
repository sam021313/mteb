from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mteb.encoder_interface import PromptType
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)


def colbert_loader(**kwargs):
    try:
        from pylate import (
            indexes as colbert_indexes,
        )
        from pylate import (
            models as colbert_model,
        )
        from pylate import (
            retrieve as colbert_retrieve,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "To use the ColBERT models `pylate` is required. Please install it with `pip install mteb[pylate]`."
        ) from e

    class ColBERTWrapper(DRESModel, Wrapper):
        def __init__(
            self,
            model_name: str,
            encode_kwargs: dict[str, Any] = {},
            corpus_chunk_size: int = 500,
            revision: str | None = None,
            model_prompts: dict[str, str] | None = None,
            **kwargs,
        ) -> None:
            """Wrapper for ColBERT models.

            Args:
                model_name: The ColBERT model to load from HuggingFace Hub.
                revision: The revision of the model to use.
                encode_kwargs: Additional arguments to pass to the encoder.
                corpus_chunk_size: The number of documents to encode at once.
                model_prompts: A dictionary mapping task names to prompt names.
                    First priority is given to the composed prompt of task name + prompt type (query or passage), then to the specific task prompt,
                    then to the composed prompt of task type + prompt type, then to the specific task type prompt,
                    and finally to the specific prompt type.
                **kwargs: Additional arguments to pass to the model.
            """
            super().__init__(None, use_sbert_model=False, **kwargs)
            self.corpus_chunk_size = corpus_chunk_size
            self.model_name = model_name
            self.encode_kwargs = encode_kwargs

            self.model_col = colbert_model.ColBERT(
                self.model_name, revision=revision, **kwargs
            )
            if (
                model_prompts is None
                and hasattr(self.model, "prompts")
                and len(self.model.prompts) > 0
            ):
                try:
                    model_prompts = self.validate_task_to_prompt_name(
                        self.model_col.prompts
                    )
                except ValueError:
                    model_prompts = None
            elif model_prompts is not None and hasattr(self.model_col, "prompts"):
                logger.info(f"Model prompts will be overwritten with {model_prompts}")
                self.model.prompts = model_prompts
            self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

        @classmethod
        def name(self):
            return "bm25s"

        def encode(
            self,
            sentences: Sequence[str],
            *,
            task_name: str,
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> np.ndarray:
            """Encodes the given sentences using the encoder.

            Args:
                sentences: The sentences to encode.
                task_name: The name of the task. Pylate uses this to
                    determine which prompt to use from a specified dictionary.
                prompt_type: The name type of prompt. (query or passage)
                **kwargs: Additional arguments to pass to the encoder.

                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)

            Returns:
                The encoded sentences as a numpy array.
            """
            prompt_name = None
            if self.model_prompts is not None:
                prompt_name = self.get_prompt_name(
                    self.model_prompts, task_name, prompt_type
                )
            if prompt_name:
                logger.info(
                    f"Using prompt_name={prompt_name} for task={task_name} prompt_type={prompt_type}"
                )
            else:
                logger.info(
                    f"No model prompts found for task={task_name} prompt_type={prompt_type}"
                )
            logger.info(f"Encoding {len(sentences)} sentences.")

            pred = self.model_col.encode(
                sentences,
                prompt_name=prompt_name,
                is_query=True if prompt_type == PromptType.query else False,
                **kwargs,
            )

            # encode returns a list of tensors shaped (x, token_dim) where x is the number of tokens in the sentence
            # we need to pad these tensors to the same length
            # Tensors have varying lengths; therefore, they need to be padded with zeros to ensure uniformity before being combined
            # output shape will be (batch_size, len(max(tokens)), embedding_token_dim)
            pred = torch.nn.utils.rnn.pad_sequence(
                pred, batch_first=True, padding_value=0
            )

            return pred.cpu().numpy()

        def search(
            self,
            corpus: dict[str, dict[str, str]],
            queries: dict[str, str | list[str]],
            top_k: int,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
            if "task_name" in kwargs:
                task_name = kwargs["task_name"]
                kwargs.pop("task_name")
            else:
                raise ValueError("task_name must be provided in kwargs")

            logger.info("Sorting Corpus by document length (Longest first)...")
            corpus_ids = sorted(
                corpus,
                reverse=True,
            )
            corpus = [corpus[cid] for cid in corpus_ids]  # type: ignore

            # Create the index
            logger.info("Encoding and indexing Corpus...")

            index_path = Path(f".cache/pylate-index/{self.model_name}-{task_name}")

            index_path.mkdir(parents=True, exist_ok=True)

            index = colbert_indexes.Voyager(
                index_folder=index_path.as_posix(),
                index_name=f"index-{self.mteb_model_meta.model_name_as_path()}-{task_name}",
                override=True,
                ef_search=top_k
                + 200,  # has to be greater than elements to be retrieved
            )

            retriever = colbert_retrieve.ColBERT(index=index)

            itr = range(0, len(corpus), self.corpus_chunk_size)
            for batch_num, corpus_start_idx in enumerate(itr):
                logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
                corpus_end_idx = min(
                    corpus_start_idx + self.corpus_chunk_size, len(corpus)
                )

                sub_corpus_embeddings = self.model_col.encode(
                    corpus[corpus_start_idx:corpus_end_idx],  # type: ignore
                    is_query=PromptType.passage,
                    **self.encode_kwargs,
                )

                index.add_documents(
                    documents_ids=corpus_ids[corpus_start_idx:corpus_end_idx],
                    documents_embeddings=sub_corpus_embeddings,
                )

            query_ids = list(queries.keys())
            self.results = {qid: {} for qid in query_ids}
            queries = [queries[qid] for qid in queries]  # type: ignore

            logger.info("Encoding Queries.")
            queries_embeddings = self.model_col.encode(
                queries,
                is_query=True,  # Encoding queries
                show_progress_bar=True,
                **self.encode_kwargs,
            )

            logger.info("Retrieving Results...")
            print(f"k: {top_k}")
            print(f"queries_embeddings len: {len(queries_embeddings)}")
            print(f"queries_embeddings[0] len: {len(queries_embeddings[0])}")
            print(f"queries_embeddings[0][0] len: {len(queries_embeddings[0][0])}")
            print(f"documents_embeddings len: {len(sub_corpus_embeddings)}")
            print(f"documents_embeddings[0] len: {len(sub_corpus_embeddings[0])}")
            print(f"documents_embeddings[0][0] len: {len(sub_corpus_embeddings[0][0])}")
            k_min = min(top_k, len(corpus))
            print(f"k_min: {k_min}")
            # print(f"index len: {len(index.ef_search.bit_count)}")
            # if "batch_size" in self.encode_kwargs:

            scores = retriever.retrieve(
                queries_embeddings=queries_embeddings,
                k=k_min,
                **self.encode_kwargs,
            )
            print(f"scores: {len(scores)}")
            for query_id, result in zip(query_ids, scores):
                self.results[query_id] = {doc["id"]: doc["score"] for doc in result}

            # Sort the results by score
            if return_sorted:
                self.results = {
                    query_id: dict(
                        sorted(docs.items(), key=lambda item: item[1], reverse=True)
                    )
                    for query_id, docs in self.results.items()
                }

            # {'PLAIN-2': {'MED-10': 8.592973709106445, 'MED-14': 8.584940910339355,
            return self.results

        def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Computes the max-similarity max_sim(a[i], b[j]) for all i and j.
                Works with a Tensor of the shape (batch_size, num_tokens, token_dim)

            Return:
                Matrix with res[i][j]  = max_sim(a[i], b[j])
            """  # noqa: D402
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)

            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, dtype=torch.float32)

            if len(a.shape) == 2:
                a = a.unsqueeze(0)

            if len(b.shape) == 2:
                b = b.unsqueeze(0)

            scores = torch.einsum(
                "ash,bth->abst",
                a,
                b,
            )

            return scores.max(axis=-1).values.sum(axis=-1)

    return ColBERTWrapper(**kwargs)


colbert_v2 = ModelMeta(
    loader=partial(colbert_loader, model_name="colbert-ir/colbertv2.0"),
    name="colbert-ir/colbertv2.0",
    languages=["eng_Latn"],
    open_weights=True,
    revision="c1e84128e85ef755c096a95bdb06b47793b13acf",
    public_training_code=True,
    release_date="2024-09-21",
    n_parameters=110 * 1e6,
    max_tokens=180,  # Reduced for Benchmarking - see ColBERT paper
    embed_dim=128,  # Bag of Embeddings (128) for each token
    license="mit",
    similarity_fn_name=None,
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    custom_search=True,
)

gercolbert = ModelMeta(
    loader=partial(
        colbert_loader, model_name="/Users/A200009373/Documents/Coding/colbert-data"
    ),
    name="gercolbert",
    languages=["eng_Latn"],
    open_weights=True,
    revision=None,
    public_training_code=True,
    release_date="2024-09-21",
    n_parameters=110 * 1e6,
    max_tokens=180,  # Reduced for Benchmarking - see ColBERT paper
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="mit",
    similarity_fn_name="max_sim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/colbert-ir/colbertv2.0",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    custom_search=True,
)


jina_colbert_v2 = ModelMeta(
    loader=partial(
        colbert_loader,
        model_name="jinaai/jina-colbert-v2",
        query_prefix="[QueryMarker]",
        document_prefix="[DocumentMarker]",
        attend_to_expansion_tokens=True,
        trust_remote_code=True,
    ),
    name="jinaai/jina-colbert-v2",
    languages=[  # list of languages the model has been evaluated on
        "ara-Arab",  # Arabic
        "ben-Beng",  # Bengali
        "deu-Latn",  # German
        "spa-Latn",  # Spanish
        "eng-Latn",  # English
        "fas-Arab",  # Persian
        "fin-Latn",  # Finnish
        "fra-Latn",  # French
        "hin-Deva",  # Hindi
        "ind-Latn",  # Indonesian
        "jpn-Jpan",  # Japanese
        "kor-Kore",  # Korean
        "rus-Cyrl",  # Russian
        "swa-Latn",  # Swahili
        "tel-Telu",  # Telugu
        "tha-Thai",  # Thai
        "yor-Latn",  # Yoruba
        "zho-Hans",  # Chinese (Simplified)
        "nld-Latn",  # Dutch
        "ita-Latn",  # Italian
        "por-Latn",  # Portuguese
        "vie-Latn",  # Vietnamese
    ],
    open_weights=True,
    revision="4cf816e5e2b03167b132a3c847a9ecd48ba708e1",
    public_training_code=False,
    release_date="2024-08-16",
    n_parameters=559 * 1e6,
    max_tokens=8192,
    embed_dim=None,  # Bag of Embeddings (128) for each token
    license="cc-by-nc-4.0",
    similarity_fn_name="max_sim",
    framework=["PyLate", "ColBERT"],
    reference="https://huggingface.co/jinaai/jina-colbert-v2",
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    custom_search=True,
)
