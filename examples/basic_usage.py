"""VRAM Fabric — Basic usage example."""

from vram_fabric import VRAMFabric


def main() -> None:
    print("=" * 60)
    print("  VRAM Fabric — Deep Tech Runtime for Local AI")
    print("=" * 60)

    # 1. Initialize the fabric
    print("\n[1] Initializing VRAM Fabric...")
    fabric = VRAMFabric(
        llm_model="llama3:8b",
        vector_dim=768,
        cache_size_mb=128,
        scheduler_policy="auto",
    )

    # 2. Index documents
    print("\n[2] Indexing documents into Vector Engine...")
    documents = [
        "A capital do Brasil é Brasília.",
        "Python é uma linguagem de programação interpretada.",
        "O Sol é uma estrela anã amarela localizada na Via Láctea.",
        "Machine learning é um subcampo da inteligência artificial.",
        "A água ferve a 100 graus Celsius ao nível do mar.",
    ]
    count = fabric.index_documents(documents, model="bge-large")
    print(f"    Indexed {count} documents.")

    # 3. Vector search
    print("\n[3] Searching vectors...")
    results = fabric.search("Qual a capital do Brasil?", k=3)
    for r in results:
        print(f"    id={r.id} score={r.score:.4f}")

    # 4. Query with semantic cache
    print("\n[4] Query (cache miss → LLM → cache insert)...")
    resp1 = fabric.query("Qual é a linguagem de programação mais popular?")
    print(f"    Response: {resp1[:120]}...")

    print("\n[4b] Query again (should be cache hit)...")
    resp2 = fabric.query("Qual é a linguagem de programação mais popular?")
    print(f"    Response: {resp2[:120]}...")

    # 5. Cache stats
    stats = fabric.cache_stats()
    print(f"\n[5] Cache stats: entries={stats.current_entries} hits={stats.hits} misses={stats.misses} "
          f"ratio={stats.hit_ratio:.2%}")

    # 6. Memory stats
    mem = fabric.memory_stats()
    print(f"\n[6] Memory: VRAM used={mem.vram_used_mb}MB free={mem.vram_free_mb}MB "
          f"spillover={mem.spillover_count}")

    # 7. Scheduler stats
    sched = fabric.scheduler_stats()
    print(f"\n[7] Scheduler: shares={sched.shares} throttled={sched.throttled}")

    # 8. Shutdown
    fabric.shutdown()
    print("\n[DONE] VRAM Fabric shut down.")


if __name__ == "__main__":
    main()
