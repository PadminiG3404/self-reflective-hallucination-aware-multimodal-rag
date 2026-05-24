"""Main pipeline orchestrator."""
from __future__ import annotations

from typing import List, Optional

import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from transformers import pipeline
from io import BytesIO

from src.encoders.multimodal_encoder import MultimodalEncoder
from src.explainability.explainer import ExplainableGenerator
from src.graph_reasoning.entity_extractor import EntityExtractor
from src.data.ingestion import build_evidence_chunks, load_dataset
from src.data.webqa import WebQAImageStore, build_webqa_evidence, load_webqa_questions
from src.graph_reasoning.semantic_graph import GraphNode, SemanticReasoningGraph
from src.hallucination.detector import HallucinationDetector
from src.refinement.retrieval_refiner import AdaptiveRetrieverRefiner
from src.reasoning.multihop_reasoner import MultiHopReasoner
from src.retrieval.faiss_retriever import CrossModalRetriever
from src.uncertainty.estimator import UncertaintyEstimator
from src.utils.schemas import EvidenceChunk, FinalExplanation
from src.utils.seed import set_seed
from src.evaluation.runner import EvaluationRunner
from src.evaluation import metrics


class HallucinationAwareMultimodalRAG:
    def __init__(self, config: dict) -> None:
        self.config = config
        set_seed(self.config["project"]["seed"])
        self.device = self.config["project"]["device"]
        self.encoder = MultimodalEncoder(
            model_type=self.config["encoder"]["model_type"],
            model_name=self.config["encoder"]["model_name"],
            device=self.device,
            normalize=self.config["encoder"]["normalize"],
        )
        self.retriever: CrossModalRetriever | None = None
        self.graph = SemanticReasoningGraph()
        self.reasoner = MultiHopReasoner(max_hops=self.config["reasoning"]["max_hops"])
        nli_pipeline = None
        if self.config["hallucination"].get("enable_nli"):
            nli_pipeline = pipeline(
                "text-classification",
                model=self.config["hallucination"]["nli_model"],
                return_all_scores=True,
            )
        self.detector = HallucinationDetector(
            similarity_threshold=self.config["hallucination"]["similarity_threshold"],
            contradiction_threshold=self.config["hallucination"]["contradiction_threshold"],
            nli_pipeline=nli_pipeline,
        )
        self.uncertainty = UncertaintyEstimator(temperature=self.config["uncertainty"]["temperature"])
        self.refiner: AdaptiveRetrieverRefiner | None = None
        self.explainer = ExplainableGenerator()
        self.entity_extractor = EntityExtractor(
            top_k=self.config.get("entities", {}).get("top_k", 3),
            min_len=self.config.get("entities", {}).get("min_len", 3),
        )
        self.repo_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.repo_root / "data"
        self.webqa_store: WebQAImageStore | None = None

    def run(self, image: Optional[object], query: str) -> FinalExplanation:
        query_embedding = self._encode_query(query, image)
        image_embedding = self._encode_image_only(image)
        evidence = self._load_evidence()
        self._ensure_index(evidence)
        top_k = self.config["retrieval"]["top_k"]
        retrieval = self.retriever.retrieve_multimodal(
            text_embedding=query_embedding,
            image_embedding=image_embedding,
            top_k=top_k,
            alpha=self.config["retrieval"].get("multimodal_alpha", 0.5),
        )

        self.graph = SemanticReasoningGraph()
        query_node = GraphNode(node_id="query", node_type="query", attributes={"text": query})
        self.graph.add_node(query_node)
        for chunk in retrieval.chunks:
            node = GraphNode(
                node_id=f"chunk_{chunk.chunk_id}",
                node_type="retrieved_chunk",
                attributes={"score": chunk.score, "source": chunk.source},
            )
            self.graph.add_node(node)
            self.graph.add_edge("query", node.node_id, relation="retrieval_dependency", weight=chunk.score)
            entities = self.entity_extractor.extract_entities(chunk.text)
            for entity in entities:
                entity_id = f"entity_{entity}"
                if entity_id not in self.graph.graph:
                    self.graph.add_node(
                        GraphNode(
                            node_id=entity_id,
                            node_type="entity",
                            attributes={"label": entity},
                        )
                    )
                self.graph.add_edge(node.node_id, entity_id, relation="semantic_relation", weight=0.5)

        reasoning_steps = self.reasoner.infer(self.graph.graph, retrieval.chunks)
        reasoning_confidence = self.reasoner.score_reasoning_path(reasoning_steps)
        graph_consistency = self.detector.graph_consistency_score(
            [step.confidence for step in reasoning_steps]
        )
        hallucination_report = self.detector.detect(
            reasoning_steps,
            retrieval_metrics=retrieval.metrics,
            graph_consistency=graph_consistency,
            evidence_texts=[chunk.text for chunk in retrieval.chunks],
        )
        uncertainty_report = self.uncertainty.estimate(
            predictive_scores=None,
            retrieval_confidence=retrieval.metrics.evidence_confidence,
            reasoning_confidence=reasoning_confidence,
        )

        if reasoning_confidence < self.config["reflection"]["refinement_confidence_threshold"]:
            retrieval = self.refiner.refine_with_text(
                query,
                retrieval.chunks,
                top_k=top_k + 2,
                graph=self.graph,
            )
            reasoning_steps = self.reasoner.infer(self.graph.graph, retrieval.chunks)
            reasoning_confidence = self.reasoner.score_reasoning_path(reasoning_steps)

        answer = "Prototype answer based on retrieved evidence."
        trace_summary = self.explainer.build_trace(reasoning_steps)
        return self.explainer.generate(
            answer=answer,
            confidence=reasoning_confidence,
            evidence_chain=retrieval.chunks,
            reasoning_steps=reasoning_steps,
            hallucination_report=hallucination_report,
            uncertainty_report=uncertainty_report,
            trace_summary=trace_summary,
        )

    def _encode_query(self, query: str, image: Optional[object]) -> np.ndarray:
        text_embeddings = self.encoder.encode_text([query]).embeddings
        if image is None:
            return text_embeddings.cpu().numpy()
        image_embeddings = self.encoder.encode_image([image]).embeddings
        combined = 0.5 * text_embeddings + 0.5 * image_embeddings
        return combined.cpu().numpy()

    def _encode_image_only(self, image: Optional[object]) -> np.ndarray | None:
        if image is None:
            return None
        image_embeddings = self.encoder.encode_image([image]).embeddings
        return image_embeddings.cpu().numpy()

    def _ensure_index(self, evidence: List[EvidenceChunk]) -> None:
        if self.retriever is not None and self.retriever.index is not None:
            return
        evidence_texts = [chunk.text for chunk in evidence]
        embeddings = self.encoder.encode_text(evidence_texts).embeddings
        embedding_dim = embeddings.shape[-1]
        self.retriever = CrossModalRetriever(embedding_dim=embedding_dim)
        image_embeddings = self._build_image_embeddings(evidence, embedding_dim)
        self.retriever.build_index(embeddings.cpu().numpy(), evidence, image_embeddings=image_embeddings)
        self.refiner = AdaptiveRetrieverRefiner(self.retriever, encoder=self.encoder)

    def _build_image_embeddings(
        self, evidence: List[EvidenceChunk], embedding_dim: int
    ) -> np.ndarray | None:
        dataset_cfg = self.config.get("dataset", {})
        has_any_images = any(
            chunk.metadata.get("image_path") or chunk.metadata.get("image_source") == "webqa_tsv"
            for chunk in evidence
        )
        if not has_any_images:
            return None
        cache: dict[str, np.ndarray] = {}
        embeddings = []
        base_dir = dataset_cfg.get("image_base_dir")
        if any(chunk.metadata.get("image_source") == "webqa_tsv" for chunk in evidence):
            tsv_path = Path(dataset_cfg.get("image_tsv"))
            if not tsv_path.is_absolute():
                tsv_path = self.repo_root / tsv_path
            target_ids = {
                str(chunk.metadata.get("image_id"))
                for chunk in evidence
                if chunk.metadata.get("image_source") == "webqa_tsv"
            }
            self.webqa_store = self.webqa_store or WebQAImageStore(
                tsv_path, target_ids=target_ids
            )
        for chunk in evidence:
            if chunk.metadata.get("image_source") == "webqa_tsv":
                image_id = chunk.metadata.get("image_id")
                cache_key = f"webqa:{image_id}"
                if cache_key in cache:
                    embeddings.append(cache[cache_key])
                    continue
                if image_id is None or self.webqa_store is None:
                    embeddings.append(np.zeros((embedding_dim,), dtype=np.float32))
                    continue
                image_bytes = self.webqa_store.get_image_bytes(str(image_id))
                if image_bytes is None:
                    embeddings.append(np.zeros((embedding_dim,), dtype=np.float32))
                    continue
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image_embedding = self.encoder.encode_image([image]).embeddings
                vector = image_embedding.cpu().numpy().reshape(-1)
                cache[cache_key] = vector
                embeddings.append(vector)
                continue
            image_path = chunk.metadata.get("image_path")
            if not image_path:
                embeddings.append(np.zeros((embedding_dim,), dtype=np.float32))
                continue
            resolved = Path(image_path)
            if not resolved.is_absolute() and base_dir:
                resolved = self.repo_root / base_dir / image_path
            cache_key = str(resolved)
            if cache_key in cache:
                embeddings.append(cache[cache_key])
                continue
            if not resolved.exists():
                embeddings.append(np.zeros((embedding_dim,), dtype=np.float32))
                continue
            image = Image.open(resolved).convert("RGB")
            image_embedding = self.encoder.encode_image([image]).embeddings
            vector = image_embedding.cpu().numpy().reshape(-1)
            cache[cache_key] = vector
            embeddings.append(vector)
        return np.stack(embeddings).astype(np.float32)

    def _load_evidence(self) -> List[EvidenceChunk]:
        dataset_cfg = self.config.get("dataset", {})
        if dataset_cfg.get("format") == "webqa":
            dataset_path = Path(dataset_cfg.get("path"))
            if not dataset_path.is_absolute():
                dataset_path = self.repo_root / dataset_path
            return build_webqa_evidence(
                dataset_path,
                split=dataset_cfg.get("split"),
                include_negatives=dataset_cfg.get("include_negatives", False),
                max_items=dataset_cfg.get("max_items"),
            )
        dataset_path_value = dataset_cfg.get("path")
        if dataset_path_value:
            dataset_path = Path(dataset_path_value)
            if not dataset_path.is_absolute():
                dataset_path = self.repo_root / dataset_path
            items = load_dataset(
                dataset_path,
                fmt=dataset_cfg.get("format"),
                max_items=dataset_cfg.get("max_items"),
            )
            if items:
                return build_evidence_chunks(
                    items,
                    chunk_size=dataset_cfg.get("chunk_size", 24),
                    overlap=dataset_cfg.get("chunk_overlap", 4),
                    source=str(dataset_path.name),
                )
        evidence_path = self.data_dir / "sample_evidence.json"
        if not evidence_path.exists():
            return self._build_dummy_evidence()
        with evidence_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [EvidenceChunk(**item) for item in raw]

    def _build_dummy_evidence(self) -> List[EvidenceChunk]:
        return [
            EvidenceChunk(chunk_id="e1", text="Sample evidence A", source="dummy", score=0.0),
            EvidenceChunk(chunk_id="e2", text="Sample evidence B", source="dummy", score=0.0),
            EvidenceChunk(chunk_id="e3", text="Sample evidence C", source="dummy", score=0.0),
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the hallucination-aware multimodal RAG pipeline")
    parser.add_argument("--query", type=str, default="What is shown in the image?")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation over dataset")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / "config" / "default.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    rag = HallucinationAwareMultimodalRAG(config=config)
    if args.evaluate:
        dataset_cfg = config.get("dataset", {})
        explanations = []
        start_times = []
        end_times = []
        eval_cfg = config.get("evaluation", {})
        checkpoint_path = Path(eval_cfg.get("checkpoint_path", "data/webqa_eval_checkpoint.json"))
        if not checkpoint_path.is_absolute():
            checkpoint_path = rag.repo_root / checkpoint_path
        checkpoint = {
            "next_index": 0,
            "count": 0,
            "sum_hallucination": 0.0,
            "sum_grounding": 0.0,
            "sum_uncertainty": 0.0,
            "sum_latency_ms": 0.0,
        }
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint.update(json.load(handle))
        log_every = int(eval_cfg.get("log_every", 25))
        start_wall = time.time()
        if dataset_cfg.get("format") == "webqa":
            dataset_path = Path(dataset_cfg.get("path"))
            if not dataset_path.is_absolute():
                dataset_path = rag.repo_root / dataset_path
            questions = load_webqa_questions(
                dataset_path,
                split=dataset_cfg.get("split"),
                max_items=eval_cfg.get("max_items"),
            )
            tsv_path = Path(dataset_cfg.get("image_tsv"))
            if not tsv_path.is_absolute():
                tsv_path = rag.repo_root / tsv_path
            rag.webqa_store = rag.webqa_store or WebQAImageStore(tsv_path)
            total_items = len(questions)
            for idx, item in enumerate(questions):
                if idx < checkpoint["next_index"]:
                    continue
                image_obj = None
                if item.get("image_id"):
                    image_bytes = rag.webqa_store.get_image_bytes(item["image_id"])
                    if image_bytes is not None:
                        image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
                start_time = time.time()
                explanation = rag.run(image=image_obj, query=item.get("question"))
                end_time = time.time()
                latency_value = metrics.latency_ms(start_time, end_time)
                hallucination_score = explanation.hallucination_report.hallucination_score
                grounding_score = 1.0 - hallucination_score
                checkpoint["count"] += 1
                checkpoint["sum_hallucination"] += hallucination_score
                checkpoint["sum_grounding"] += grounding_score
                checkpoint["sum_uncertainty"] += explanation.uncertainty_report.calibration_score
                checkpoint["sum_latency_ms"] += latency_value
                checkpoint["next_index"] = idx + 1
                if log_every and checkpoint["count"] % log_every == 0:
                    elapsed = time.time() - start_wall
                    remaining = total_items - checkpoint["next_index"]
                    rate = checkpoint["count"] / max(1.0, elapsed)
                    eta = remaining / max(rate, 1e-6)
                    print(
                        f"Processed {checkpoint['next_index']}/{total_items} items "
                        f"({rate:.2f} items/s, ETA {eta/60:.1f} min)"
                    )
                    with checkpoint_path.open("w", encoding="utf-8") as handle:
                        json.dump(checkpoint, handle)
        else:
            items = load_dataset(
                Path(dataset_cfg.get("path")),
                fmt=dataset_cfg.get("format"),
                max_items=eval_cfg.get("max_items"),
            )
            total_items = len(items)
            for idx, item in enumerate(items):
                if idx < checkpoint["next_index"]:
                    continue
                image_obj = None
                if item.image_path:
                    image_path = Path(item.image_path)
                    if not image_path.is_absolute():
                        image_path = rag.repo_root / dataset_cfg.get("image_base_dir", "data") / item.image_path
                    if image_path.exists():
                        image_obj = Image.open(image_path).convert("RGB")
                start_time = time.time()
                explanation = rag.run(image=image_obj, query=item.text)
                end_time = time.time()
                latency_value = metrics.latency_ms(start_time, end_time)
                hallucination_score = explanation.hallucination_report.hallucination_score
                grounding_score = 1.0 - hallucination_score
                checkpoint["count"] += 1
                checkpoint["sum_hallucination"] += hallucination_score
                checkpoint["sum_grounding"] += grounding_score
                checkpoint["sum_uncertainty"] += explanation.uncertainty_report.calibration_score
                checkpoint["sum_latency_ms"] += latency_value
                checkpoint["next_index"] = idx + 1
                if log_every and checkpoint["count"] % log_every == 0:
                    elapsed = time.time() - start_wall
                    remaining = total_items - checkpoint["next_index"]
                    rate = checkpoint["count"] / max(1.0, elapsed)
                    eta = remaining / max(rate, 1e-6)
                    print(
                        f"Processed {checkpoint['next_index']}/{total_items} items "
                        f"({rate:.2f} items/s, ETA {eta/60:.1f} min)"
                    )
                    with checkpoint_path.open("w", encoding="utf-8") as handle:
                        json.dump(checkpoint, handle)
        if checkpoint["count"] == 0:
            print("No evaluation items were processed.")
        else:
            summary = {
                "hallucination_score": checkpoint["sum_hallucination"] / checkpoint["count"],
                "grounding_score": checkpoint["sum_grounding"] / checkpoint["count"],
                "uncertainty": checkpoint["sum_uncertainty"] / checkpoint["count"],
                "latency_ms": checkpoint["sum_latency_ms"] / checkpoint["count"],
                "count": checkpoint["count"],
            }
            checkpoint["summary"] = summary
            with checkpoint_path.open("w", encoding="utf-8") as handle:
                json.dump(checkpoint, handle)
            print(json.dumps({"summary": summary}, indent=2))
    else:
        image_obj = Image.open(args.image).convert("RGB") if args.image else None
        output = rag.run(image=image_obj, query=args.query)
        print(output.model_dump_json(indent=2))
