# PRD - VRAM Fabric v1.0 (Deep Tech Runtime)

**Product Requirements Document**
**Versão**: 1.0 | **Status**: Draft | **Data**: 2026-06-23
**Autor**: HPC Systems Architect | **Stakeholders**: AI Engineers, MLOps, Research Labs

---

## 1. Resumo Executivo e Visão de Produto

### 1.1 O Problema Físico

Toda stack de IA local sofre do mesmo gargalo: **o barramento PCIe**. Enquanto a VRAM GDDR6/X opera a ~600-1000 GB/s, o PCIe 4.0 x16 entrega apenas ~32 GB/s — uma diferença de **18-30x**. Cada vez que um embedding, vetor ou estado de agente transita CPU↔GPU, o sistema paga o "PCIe tax": latência de dezenas de microssegundos e energia desperdiçada em transferências de DMA.

### 1.2 A Revolução

O **VRAM Fabric** é um *runtime operating system para IA local* — não uma aplicação, mas a fundação sobre a qual aplicações rodam. Ele transforma a VRAM no barramento principal de dados, tratando a RAM do sistema como storage secundário (spillover), não como memória primária.

Toda a stack — vetores, embeddings, cache semântico, estado de agentes, filas de execução — reside em VRAM, acessada por ponteiros de dispositivo CUDA com zero-copy entre contextos.

### 1.3 Posicionamento

| Característica | API Gateway | Scheduler | VRAM Fabric |
|---|---|---|---|
| Onde os dados residem | RAM | RAM | **VRAM** |
| Latência acesso a vetores | 50-200µs (PCIe) | 50-200µs | **<5µs (HBM)** |
| Cache semântico | Redis (RAM) | Redis (RAM) | **VRAM kernel CUDA** |
| Multiplicador | 1x | 1x | **18-30x** |

O VRAM Fabric é um **multiplicador de força**: toda stack construída sobre ele herda a redução de latência automaticamente.

---

## 2. Análise da Dor Atual

### 2.1 Quantificação do Overhead CPU-GPU

| Operação | Tradicional (CPU RAM + GPU) | VRAM Fabric | Redução |
|---|---|---|---|
| Busca vetorial 1M (batch 1) | 15-50ms (inclui PCIe) | <5ms | **3-10x** |
| Troca de contexto entre agentes | 10-100ms (serialização) | <1ms (zero-copy tensor) | **10-100x** |
| Cache hit semantic | 0.5-2ms (Redis/RAM) | <100µs (VRAM kernel) | **5-20x** |
| Overhead de scheduler | 5-15% GPU idle | <2% | **2.5-7.5x** |
| Energia por query | ~50µJ (PCIe) | ~5µJ (on-die) | **10x** |

### 2.2 O Custo do PCIe Round-trip

```
Tradicional:
  CPU → malloc RAM → copy to GPU (PCIe 32 GB/s) → kernel → copy back (PCIe) → CPU

VRAM Fabric:
  GPU → cudaMalloc VRAM → kernel (600 GB/s) → resultado em VRAM
```

---

## 3. Escopo do Projeto

### IN SCOPE

| Item | Descrição |
|---|---|
| Vector Engine | Indexação plana + IVF 100% CUDA (FAISS GPU + CuPy) |
| Agent Runtime | Modelo de atores com estado em tensores VRAM, zero-copy context switch |
| Semantic Cache | LRU gerenciado por kernel CUDA, hit detection via dot product batch |
| Dynamic Scheduler | Feedback loop baseado em SM occupancy e VRAM pressure |
| Memory Pool | Alocador de VRAM com spillover automático RAM→SSD |
| SDK Python | API declarativa: `fabric.query()`, `fabric.register_agent()` |
| Telemetria | Métricas de SM occupancy, VRAM bandwidth utilization, cache hit ratio |

### OUT OF SCOPE

| Item | Justificativa |
|---|---|
| Multi-node distribuído | Foco single-node multi-GPU na v1 |
| Treinamento de modelos | Inference-only runtime |
| Interface gráfica | Headless, SDK Python apenas |
| Suporte a GPU AMD/Intel | NVIDIA CUDA first |

---

## 4. Personas e Casos de Uso

### 4.1 Pesquisador de IA (Dr. Silva)
Executa experimentos com RAG + agentes locais no Llama 3 70B com 24GB VRAM.
- **Uso**: `fabric = VRAMFabric()` → `fabric.index_documents(docs)` → `fabric.query("análise do gene BRCA1")`
- **Benefício**: Resultados em <100ms vs 500ms+ da stack tradicional.

### 4.2 Engenheiro de MLOps (João)
Gerencia pipeline de inferência com múltiplos modelos e cache.
- **Uso**: `fabric.set_scheduler_policy(llm=0.70, vectors=0.20, cache=0.10)`
- **Benefício**: 3x mais throughput com mesma GPU, zero código de gerenciamento de memória.

### 4.3 Desenvolvedor de Agentes (Clara)
Constrói sistema multi-agente com 5 sub-agentes em paralelo.
- **Uso**: `fabric.spawn_agent("analyst")`, `fabric.spawn_agent("critic")`, `fabric.run_workflow()`
- **Benefício**: Context switch entre agentes em <1ms, compartilhamento de embeddings zero-copy.

---

## 5. Arquitetura Técnica Detalhada

### 5.1 Diagrama de Fluxo (ASCII)

```
                    ┌──────────────────────────────────────────────┐
                    │              VRAM FABRIC RUNTIME              │
                    │                                              │
  User Prompt ─────►│  API Server (FastAPI + uvloop)               │
                    │         │                                    │
                    │         ▼                                    │
                    │  ┌──────────────────────────────────────┐    │
                    │  │        DYNAMIC SCHEDULER              │    │
                    │  │  ┌────┐ ┌────┐ ┌────┐ ┌──────────┐  │    │
                    │  │  │LLM │ │Vec │ │Agt │ │Cache     │  │    │
                    │  │  │70% │ │15% │ │10% │ │5%        │  │    │
                    │  │  └────┘ └────┘ └────┘ └──────────┘  │    │
                    │  └──────────────┬───────────────────────┘    │
                    │                 │                             │
                    │    ┌────────────┼────────────┐               │
                    │    ▼            ▼            ▼               │
                    │ ┌──────┐  ┌──────────┐  ┌─────────┐        │
                    │ │Vector│  │ Agent     │  │Semantic │        │
                    │ │Engine│  │ Runtime   │  │Cache    │        │
                    │ │      │  │           │  │         │        │
                    │ │FAISS │  │PyTorch    │  │CUDA LRU │        │
                    │ │GPU   │  │Tensors    │  │Kernel   │        │
                    │ └──┬───┘  └─────┬─────┘  └────┬────┘        │
                    │    │             │              │            │
                    │    └─────────────┼──────────────┘            │
                    │                  │                           │
                    │    ┌─────────────┴──────────────┐            │
                    │    │     MEMORY POOL (VRAM)      │            │
                    │    │  ┌──────────────────────┐   │            │
                    │    │  │ cudaMalloc arena     │   │            │
                    │    │  │ (24GB unified pool)  │   │            │
                    │    │  └──────────────────────┘   │            │
                    │    │  Spillover: VRAM→RAM→SSD     │            │
                    │    └─────────────────────────────┘            │
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                                    Local LLM
                              (llama.cpp / vLLM)
```

### 5.2 Modelo de Memória Unificada

```
┌─────────────────────────────────────────────────────────┐
│                    VRAM (24GB typical)                    │
│  ┌──────────┬──────────┬──────────┬───────────────────┐ │
│  │ LLM      │ Vector   │ Agent    │ Cache + Free      │ │
│  │ Weights  │ Index    │ States   │ Pool              │ │
│  │ 14GB     │ 3GB      │ 2GB      │ 5GB               │ │
│  └──────────┴──────────┴──────────┴───────────────────┘ │
│                         ↕ spillover                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │              RAM (32GB) — Cold Storage             │   │
│  │  Evicted vectors, frozen agent states, old cache  │   │
│  └──────────────────────────────────────────────────┘   │
│                         ↕ spillover                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SSD — Archive Tier                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

Estratégia: alocação explícita via `cudaMalloc` + `torch.cuda.empty_cache()` gerenciada pelo Memory Pool. Nada de UVM (Unified Virtual Memory) automática — o page-fault driver do UVM tem latência de 10-50µs, anulando o benefício da VRAM. Em vez disso, o spillover é manual e preditivo, baseado em heurística de acesso futuro (LRU com lookahead).

---

## 6. Especificação dos Componentes (Requisitos Funcionais)

### RF01 — Vector Engine

| Campo | Especificação |
|---|---|
| **Índices** | Flat (brute-force via cuBLAS batched matmul), IVF (inverted file via FAISS GPU) |
| **Métrica** | Inner product (dot), L2, cosine |
| **Batch ANN** | Multi-query em paralelo via CUDA streams independentes |
| **Fallback CPU** | Spillover automático: quando VRAM < 512MB free, evicta shards IVF para RAM, reconstrói sob demanda |
| **API** | `engine.add(vectors)`, `engine.search(query, k=10)`, `engine.remove(ids)` |
| **Precision** | FP16 por padrão, FP32 opcional para sensibilidade |

### RF02 — Agent Runtime

| Campo | Especificação |
|---|---|
| **Modelo** | Atores (actors) no estilo Erlang/OTP, cada agente = 1 tensor de estado + 1 CUDA stream |
| **Estado** | Tensor `[hidden_dim]` alocado em VRAM via `torch.zeros(..., device='cuda')` |
| **Context Switch** | Swap de ponteiro de tensor: O(1), < 1µs. Sem cópia CPU↔GPU |
| **Isolamento** | Streams CUDA separadas (`cudaStream_t`) garantem que kernels de agentes distintos não interfiram |
| **Workflow** | DAG de tarefas com dependências: `Task A → Task B → Task C` executado no mesmo stream para localidade |
| **API** | `agent = runtime.spawn(name, model, prompt)`, `runtime.step(agent_id)`, `runtime.collect()` |

### RF03 — Semantic Cache

| Campo | Especificação |
|---|---|
| **Armazenamento** | Triplas `{query_hash, embedding_fp16, response_tokens}` em buffer circular VRAM |
| **Política** | LRU com timestamp atômico (contador de ciclo GPU) |
| **Hit Detection** | Batch dot-product: `query_emb @ cache_embs.T → argmax(similarity > threshold)` |
| **Kernel CUDA** | Kernel customizado `lru_evict_and_insert_kernel<<<grid, block>>>` que faz eviction + insertion em 1 kernel launch |
| **Threshold** | Similaridade cosseno > 0.95 → cache hit |
| **API** | `cache.lookup(embedding) → response | None`, `cache.insert(query_emb, response)` |

### RF04 — Dynamic Scheduler

| Campo | Especificação |
|---|---|
| **Algoritmo** | Proportional-Integral (PI) controller: mede SM occupancy de cada engine, ajusta frações |
| **Métrica** | `sm_occupancy_ratio = active_warps / max_warps_per_sm` |
| **Feedback** | A cada 100ms: `if vector_occupancy > 0.8: vector_share += 0.05` |
| **Limites** | LLM: 50-80%, Vector: 5-30%, Agent: 5-25%, Cache: 2-15% |
| **Preempção** | Kernel de longa duração (LLM) não preemptado — ajuste ocorre entre gerações de token |
| **API** | `scheduler.set_policy({...})`, `scheduler.get_utilization()` |

---

## 7. Requisitos Não-Funcionais

| RNF | Métrica | Prioridade |
|---|---|---|
| **RNF01** — Latência busca vetorial | <5ms para 1M vetores (batch 1, FP16) | P0 |
| **RNF02** — Throughput cache | >10.000 queries/s | P0 |
| **RNF03** — Overhead scheduler | <2% de ciclos de GPU | P0 |
| **RNF04** — VRAM mínima | 8GB (desktop: RTX 3070+), 24GB (workstation: RTX 4090) | P0 |
| **RNF05** — Context switch agente | <1ms | P1 |
| **RNF06** — Spillover latency | <50ms para recover de RAM, <500ms de SSD | P1 |
| **RNF07** — Cold start | <5s para inicializar engines (sem LLM) | P2 |
| **RNF08** — Precisão numérica | Erro <1% vs FP32 para cosine similarity em FP16 | P2 |
| **RNF09** — Suporte concorrência | 16 CUDA streams paralelas, múltiplos agentes simultâneos | P1 |

---

## 8. Modelo de Dados e API Pública

### 8.1 Inicialização

```python
from vram_fabric import VRAMFabric

# Inicializa com detecção automática de VRAM disponível
fabric = VRAMFabric(
    llm_model="llama3:8b",       # via llama.cpp
    vector_dim=4096,             # dimensão dos embeddings
    cache_size_mb=2048,          # 2GB para cache semântico
    scheduler_policy="auto",     # "auto" | "balanced" | dict manual
)

# Registra documentos para busca vetorial
fabric.vector_engine.index(documents=my_docs, model="bge-large")

# Registra agentes
fabric.register_agent("assistant", system_prompt="Você é um assistente útil.")
fabric.register_agent("critic", system_prompt="Você analisa criticamente respostas.")

# Query unificada — o fabric decide cache vs LLM vs agente
response = fabric.query("Qual a capital do Brasil?")
# → Cache hit? Retorna instantaneamente.
# → Cache miss? Roda LLM + atualiza cache + retorna resposta.

# Workflow multi-agente
workflow = fabric.create_workflow([
    fabric.task("assistant", "Analise o documento X"),
    fabric.task("critic", "Revise a análise do assistant"),
])
result = fabric.run(workflow)
```

### 8.2 API Declarativa Completa

```python
class VRAMFabric:
    # Vector Engine
    def index_documents(docs, model) -> IndexHandle
    def search(query, k=10) -> list[SearchResult]
    def search_batch(queries, k=10) -> list[list[SearchResult]]

    # Agent Runtime
    def register_agent(name, system_prompt, model=None) -> AgentHandle
    def spawn_agent(name) -> AgentInstance
    def task(agent, prompt) -> Task
    def create_workflow(tasks) -> Workflow
    def run(workflow) -> WorkflowResult

    # Cache
    def cache_stats() -> CacheStats  # size, hits, misses, hit_ratio

    # Scheduler
    def set_scheduler_policy(policy) -> None
    def scheduler_stats() -> SchedulerStats

    # Memory
    def memory_stats() -> MemoryStats  # vram_used, vram_free, spillover_count

    # Lifecycle
    def shutdown() -> None
```

---

## 9. Estratégia de Falhas e Degradação Graciosa

### 9.1 VRAM Lotação

```
┌──────────────────────────────────────────────────┐
│              SPILLOVER TRIGGERS                   │
│                                                   │
│  VRAM free < 512MB ──► Evict LRU vectors → RAM   │
│  VRAM free < 256MB ──► Evict cold cache → RAM    │
│  VRAM free < 128MB ──► Freeze agent → RAM        │
│  VRAM free < 64MB  ──► Emergency: swap to SSD    │
│                                                   │
│  Recovery: quando VRAM free > 1GB, reload da RAM │
└──────────────────────────────────────────────────┘
```

### 9.2 GPU OOM (Out of Memory)

- Capturar `cudaErrorMemoryAllocation`
- Tentar alocação com tamanho reduzido (batch splitting)
- Se impossível, reportar erro gracioso (nunca crashar o processo host)

### 9.3 Falha de Kernel CUDA

- Cada engine opera em stream CUDA isolado
- Timeout watchdog: se kernel > 5s sem completar, aborta stream e reinicia engine
- Erro de kernel não contamina outras engines (isolamento por stream)

---

## 10. Roadmap de Implementação

### Fase 1 — MVP (Semanas 1-4): "Vector + Cache"

| Semana | Entregável |
|---|---|
| S1 | Memory Pool + alocador VRAM + configuração YAML |
| S2 | Vector Engine com FAISS GPU (índice plano + IVF) + API Python |
| S3 | Semantic Cache com LRU + kernel CUDA de hit detection |
| S4 | Testes de benchmark (1M vetores, 10K cache queries) + CLI básico |

### Fase 2 — Evolução (Semanas 5-8): "Agents + Scheduler"

| Semana | Entregável |
|---|---|
| S5 | Agent Runtime com tensores de estado em VRAM + zero-copy context switch |
| S6 | Workflow DAG engine + execução paralela de agentes |
| S7 | Dynamic Scheduler com PI controller + feedback de SM occupancy |
| S8 | Spillover RAM/SSD + recuperação automática |

### Fase 3 — Maturidade (Semanas 9-12): "Produção + Otimização"

| Semana | Entregável |
|---|---|
| S9 | Integração com Triton Inference Server para kernel fusion |
| S10 | Suporte multi-GPU (NCCL all-reduce para índices distribuídos) |
| S11 | API Server REST + WebSocket (FastAPI) para acesso remoto |
| S12 | Benchmark suite, profiling com Nsight, otimização de occupancy |

---

## 11. Métricas de Sucesso (KPIs)

| KPI | Alvo | Medição |
|---|---|---|
| **KPI-01** Redução latência end-to-end vs tradicional | >5x | Comparação A/B com stack CPU+GPU |
| **KPI-02** Cache hit ratio | >60% | `hits / (hits + misses)` |
| **KPI-03** VRAM bandwidth utilization | >80% | `bytes_read_written / (time * max_bandwidth)` |
| **KPI-04** SM occupancy média | >60% | `active_warps / max_warps` via nvml |
| **KPI-05** Latência busca vetorial p99 | <10ms | Benchmark com 1M vetores FP16 |
| **KPI-06** Overhead do scheduler | <2% | `(scheduler_kernel_time / total_gpu_time) * 100` |
| **KPI-07** Throughput cache | >10K qps | `queries / second` em batch de 1000 |

---

## 12. Diferencial Competitivo

### 12.1 Comparação Objetiva

| Solução | VRAM residente? | Cache em GPU? | Zero-copy agents? | Scheduler GPU-native? |
|---|---|---|---|---|
| AI API Gateway | Não | Não | Não | Não |
| GPU Task Scheduler | Parcial | Não | Não | Sim |
| Local AI Cluster | Não | Não | Não | Não |
| Model Router | Não | Não | Não | Não |
| AI Observability | Não | Não | Não | Não |
| **VRAM Fabric** | **Sim** | **Sim** | **Sim** | **Sim** |

### 12.2 Multiplicador de Força

O VRAM Fabric é um **"force multiplier"**: qualquer stack construída sobre ele (API Gateway, Router, Observability) herda automaticamente:

- **Latência 18-30x menor** (VRAM vs PCIe)
- **Cache semântico gratuito** (todo query é potencial cache hit)
- **Zero-copy entre componentes** (vetores, agentes, prompts — todos no mesmo address space CUDA)

Isso não é "melhor que a concorrência" — é **ortogonal** a ela. O VRAM Fabric é a camada abaixo.

---

## Código Conceitual

### Scheduler PI Controller (pseudocódigo)

```python
class DynamicScheduler:
    def __init__(self):
        self.shares = {"llm": 0.70, "vector": 0.15, "agent": 0.10, "cache": 0.05}
        self.target_occupancy = 0.70

    def tick(self, occupancy: dict[str, float]):
        """Feedback loop a cada 100ms baseado em SM occupancy."""
        for engine, target_share in self.shares.items():
            error = self.target_occupancy - occupancy.get(engine, 0)
            # PI controller
            adjustment = 0.05 * error  # Kp = 0.05
            self.shares[engine] = clamp(target_share + adjustment, 0.02, 0.80)
        self._normalize_shares()
        self._apply_cuda_mempool_limits()
```

### Cache LRU Kernel (pseudocódigo CUDA)

```cuda
__global__ void lru_evict_and_insert(
    half* cache_embeddings,  // [max_entries, dim]
    int* cache_timestamps,   // [max_entries]
    half* new_embedding,     // [dim]
    int current_cycle
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        // Find LRU entry (oldest timestamp)
        int lru_idx = 0;
        int min_ts = cache_timestamps[0];
        for (int i = 1; i < MAX_ENTRIES; i++) {
            if (cache_timestamps[i] < min_ts) {
                min_ts = cache_timestamps[i];
                lru_idx = i;
            }
        }
        // Evict + insert in place
        int offset = lru_idx * DIM;
        for (int d = 0; d < DIM; d++) {
            cache_embeddings[offset + d] = new_embedding[d];
        }
        cache_timestamps[lru_idx] = current_cycle;
    }
}
```

---

## 13. Perguntas em Aberto (Open Questions)

1. **FP16 vs INT8 para embeddings no cache**: INT8 reduz VRAM em 2x, mas cosine similarity perde ~0.5% de precisão. O threshold de cache (0.95) precisa ser recalibrado. Vale o trade-off?
2. **Preempção de kernel LLM**: CUDA não suporta preempção de kernel em user space. Entre gerações de token, o scheduler pode redimensionar. Mas se o LLM ocupa 80% e o Vector Engine precisa de mais VRAM, devemos bloquear novas queries ou evictar o índice IVF parcialmente?
3. **Suporte AMD ROCm**: A stack é CUDA-first. O esforço para portar para HIP/ROCm é ~20% do código (FAISS e PyTorch já suportam). Deve entrar no roadmap da v1 ou v2?
4. **Multi-GPU com NCCL**: Distribuir o índice vetorial via all-reduce NCCL aumenta throughput mas adiciona latência de comunicação inter-GPU (~10µs por NVLink). O ganho compensa para índices >10M vetores?
5. **Segurança entre tenants na mesma GPU**: MIG (Multi-Instance GPU) em A100/H100 isola VRAM e SMs fisicamente. Em GPUs consumer (RTX), o isolamento é apenas lógico (streams CUDA). Isso é suficiente para workloads multi-tenant?
6. **Cold start do cache**: Pré-aquecer o cache com queries sintéticas baseadas no domínio do usuário (ex: carregar 1000 perguntas frequentes) reduziria o tempo até atingir 60% hit ratio?
