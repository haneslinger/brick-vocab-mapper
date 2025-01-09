from functools import cached_property
from pathlib import Path
from tqdm import tqdm
from loguru import logger as log
import os
from fnvhash import fnv, fnv1a_32
import numpy as np
from collections import defaultdict
from usearch.index import Index
import openai
from typing import Optional, List, Dict, Iterable
from rdflib import Graph, BRICK
from rdflib.term import Node
from dataclasses import dataclass, field
from pprint import pprint
import ollama
import chromadb

class Mapper:
    def __init__(
        self,
        first_definitions: list[dict],
        first_index_file: Path,
        second_definitions: list[dict],
        second_index_file: Path,
    ):
        self.first_definitions = first_definitions
        self.first_index = self.populate_external_embeddings(first_definitions, first_index_file)

        self.second_definitions = second_definitions
        self.second_index = self.populate_external_embeddings(second_definitions, second_index_file)

    @classmethod
    def get_embedding(cls, text: str) -> np.ndarray:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding = response["embedding"]
    
        return np.array(embedding)  # type: ignore

    @classmethod
    def populate_external_embeddings(cls, definitions: list[dict], index_file: str):
        """
        Calculate embeddings for the definitions passed in. Expect ['name'] key
        """
        index = Index(ndim=1024, metric="cos")

        if Path(index_file).exists():
            log.info(f"Restoring {index_file}...")
            index.load(index_file)

            return index
        
        log.info(f"{index_file} not found. Computing embeddings for external concepts")
        for i, defn in enumerate(tqdm(definitions)):
            name = defn.pop("name")
            values = filter(lambda x: x and len(x), defn.values())
            text = f"{name} {' '.join(values)}".strip()

            index.add(i, cls.get_embedding(text))
        
        index.save(index_file)
        return index
    
    def get_one_to_one_mappings(self) -> Dict[str, Node]:
        join = self.first_index.join(self.second_index)
        return {
            str(self.first_definitions[i]): self.second_definitions[j]
            for i, j in join.items()
        }
    
    def get_mappings_with_collisions(self, top_k=3, threshold=0.8) -> Dict[Node, List[dict]]: 
        mapping = {}

        for key, first_definition in enumerate(self.first_definitions):
            name = first_definition["name"]
            embedding = self.first_index.get(key)

            recommendations = self.second_index.search(embedding, count=top_k)
            best_recommendations = [r for r in recommendations if r.distance < threshold]

            mapping[name] = [(self.second_definitions[r.key]["name"], r.distance) for r in best_recommendations]

        return mapping