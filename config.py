# -*- coding: utf-8 -*-
"""Configurações centralizadas do Pipeline ARES.

Todas as constantes, pesos, chaves de API e paths são definidos aqui.
Nenhum valor deve ser hardcoded nos demais módulos.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuração principal do pipeline de avaliação ARES.

    Attributes:
        TOP_K: Número de artigos no ranking final.
        QUALI_WEIGHT: Peso do escore qualitativo no Score Global.
        QUANT_WEIGHT: Peso do escore quantitativo no Score Global.
        C_WEIGHT: Peso das citações brutas no S_Quant.
        V_WEIGHT: Peso das citações influentes no S_Quant.
        NOTA_CORTE: Nota mínima de S_Quali para aprovação.
        LIMITE_COLETA: Número máximo de artigos a coletar.
        MODO_PRODUCAO: Se False, usa mocks para LLMs (sem gastar créditos).
        EMBEDDING_MODEL: Modelo padrão de SentenceTransformer.
        EMBEDDING_FALLBACK: Modelo fallback se o principal falhar.
        LLM_TEMPERATURE: Temperatura dos LLMs (0 = determinístico).
        LLM_MODELS: Lista de modelos a usar no ensemble.
    """

    # Scores e pesos
    TOP_K: int = 5
    QUALI_WEIGHT: float = 0.6
    QUANT_WEIGHT: float = 0.4
    C_WEIGHT: float = 0.7
    V_WEIGHT: float = 0.3
    NOTA_CORTE: float = 5.0

    # Coleta
    LIMITE_COLETA: int = 500
    ANOS_COLETA: list[int] = field(default_factory=lambda: [2022, 2023, 2024, 2025])

    # Modo
    MODO_PRODUCAO: bool = True

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_FALLBACK: str = "all-mpnet-base-v2"

    # LLMs
    LLM_TEMPERATURE: float = 0.0
    LLM_MODELS: list[str] = field(
        default_factory=lambda: ["qwen", "gemma3", "llama3.2"]
    )

    # Modelos específicos de cada provider
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GROK_MODEL: str = "grok-2-latest"

    # Ollama (local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODELS: dict[str, str] = field(
        default_factory=lambda: {
            "qwen": "qwen2.5:3b",
            "gemma3": "gemma3",
            "llama3.2": "llama3.2:3b",
        }
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    PROMPT_DIR: Path = BASE_DIR / "prompts"
    CACHE_DIR: Path = BASE_DIR / "cache"
    RESPONSES_DIR: Path = CACHE_DIR / "responses"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # Arquivos de output
    DATASET_RAW: Path = OUTPUT_DIR / "dataset_raw.csv"
    DATASET_PREPROCESSED: Path = OUTPUT_DIR / "dataset_preprocessed.csv"
    DATASET_EMBEDDINGS: Path = OUTPUT_DIR / "dataset_embeddings.pkl"
    AGREEMENT_CSV: Path = OUTPUT_DIR / "agreement.csv"
    RANKING_CSV: Path = OUTPUT_DIR / "ranking.csv"
    DATASET_FINAL_XLSX: Path = OUTPUT_DIR / "dataset_final.xlsx"
    EXPERIMENT_JSON: Path = OUTPUT_DIR / "experiment.json"
    PIPELINE_LOG: Path = OUTPUT_DIR / "pipeline.log"

    # Cache
    EMBEDDINGS_CACHE: Path = CACHE_DIR / "embeddings.pkl"

    # Prompt files
    PROMPT_RELEVANCE: Path = PROMPT_DIR / "relevance.txt"
    PROMPT_SEMANTIC_PROFILE: Path = PROMPT_DIR / "semantic_profile.txt"
    PROMPT_RELEVANCE_EMBED: Path = PROMPT_DIR / "relevance_embed.txt"

    # Cache de respostas com embeddings (separado do cache sem embeddings)
    RESPONSES_EMBED_DIR: Path = CACHE_DIR / "responses_embed"


# Instância singleton de configuração
config = PipelineConfig()

# ---------------------------------------------------------------------------
# API Keys (via variáveis de ambiente — nunca hardcoded)
# ---------------------------------------------------------------------------
API_KEYS: dict[str, str] = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "grok": os.getenv("GROK_API_KEY", ""),
}

# ---------------------------------------------------------------------------
# User-Agent para requisições HTTP (politeness)
# ---------------------------------------------------------------------------
USER_AGENT: str = "ARES-Auditor/2.0 (auditoria.academica@universidade.edu)"

# ---------------------------------------------------------------------------
# Termos de pesquisa para DBLP (Fase 1 — preservado)
# ---------------------------------------------------------------------------
TERMOS_PESQUISA: list[str] = [
    "Blockchain",
    "IoT",
    "Cyber",
    "Fault",
    "Distributed",
    "Resilience",
]

# ---------------------------------------------------------------------------
# Domínios para o prompt de relevância
# ---------------------------------------------------------------------------
DOMAINS: list[str] = [
    "CPS",
    "Blockchain",
    "IoT",
    "Fault Tolerance",
    "Distributed Systems",
]


# ---------------------------------------------------------------------------
# Descrições dos domínios de pesquisa (para embeddings de domínio)
# ---------------------------------------------------------------------------

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "CPS": (
        "Cyber-Physical Systems (CPS) are engineered systems that are built from, "
        "and depend upon, the seamless integration of computation and physical components. "
        "Advances in CPS will enable capability, adaptability, scalability, resiliency, "
        "safety, security, and usability that will expand the horizons of these critical systems. "
        "CPS technology transforms how people interact with engineered systems in transportation, "
        "healthcare, manufacturing, energy, agriculture, and defense."
    ),
    "Blockchain": (
        "Blockchain is a distributed, decentralized, immutable ledger technology that enables "
        "secure and transparent peer-to-peer transactions without intermediaries. "
        "Key concepts include smart contracts, consensus mechanisms (Proof of Work, Proof of Stake, "
        "PBFT), distributed ledger technology (DLT), cryptographic hashing, Merkle trees, "
        "and decentralized applications (dApps). Applications span finance (DeFi), "
        "supply chain traceability, identity management, and secure data sharing."
    ),
    "IoT": (
        "Internet of Things (IoT) refers to the network of physical objects embedded with sensors, "
        "software, and connectivity that enables them to collect and exchange data. "
        "Key topics include sensor networks, edge computing, MQTT/CoAP protocols, "
        "device interoperability, energy harvesting, low-power wireless communication, "
        "and real-time data processing. IoT applications include smart homes, industrial IoT (IIoT), "
        "smart cities, healthcare monitoring, and environmental sensing."
    ),
    "Fault Tolerance": (
        "Fault tolerance is the ability of a system to continue operating properly in the event "
        "of the failure of some of its components. Key concepts include redundancy (hardware, "
        "software, information, time), Byzantine fault tolerance, checkpointing and rollback "
        "recovery, replication protocols, consensus in distributed systems, failure detection, "
        "self-stabilization, and graceful degradation. Critical for safety-critical systems, "
        "avionics, nuclear power, medical devices, and financial infrastructure."
    ),
    "Distributed Systems": (
        "Distributed systems are collections of independent computers that appear to users "
        "as a single coherent system. Key topics include consensus algorithms (Raft, Paxos), "
        "distributed databases and storage, CAP theorem, consistency models, replication "
        "and sharding, distributed coordination (ZooKeeper, etcd), message passing, "
        "microservices architecture, container orchestration (Kubernetes), cloud computing, "
        "and edge-cloud continuum."
    ),
}
