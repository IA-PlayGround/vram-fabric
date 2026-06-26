"""VRAM Fabric — Multi-agent workflow example."""

from vram_fabric import VRAMFabric


def main() -> None:
    print("=" * 60)
    print("  VRAM Fabric — Multi-Agent Workflow")
    print("=" * 60)

    fabric = VRAMFabric(vector_dim=768, cache_size_mb=128)

    # Register agents
    print("\n[1] Registering agents...")
    fabric.register_agent("researcher", "Você é um pesquisador. Analise dados e forneça fatos.")
    fabric.register_agent("critic", "Você é um crítico. Revise o trabalho do pesquisador e sugira melhorias.")
    fabric.register_agent("summarizer", "Você é um sumarizador. Resuma as conclusões de forma concisa.")
    print("    Agents registered: researcher, critic, summarizer")

    # Create workflow: researcher → critic → summarizer
    print("\n[2] Creating DAG workflow...")
    t1 = fabric.task("researcher", "Quais são os principais frameworks de deep learning em 2026?")
    t1.task_id = "t1"

    t2 = fabric.task("critic", "Revise a pesquisa sobre frameworks de deep learning.")
    t2.task_id = "t2"
    t2.dependencies = ["t1"]

    t3 = fabric.task("summarizer", "Resuma as conclusões sobre frameworks de deep learning.")
    t3.task_id = "t3"
    t3.dependencies = ["t2"]

    workflow = fabric.create_workflow([t1, t2, t3])

    # Run
    print("\n[3] Running workflow...")
    result = fabric.run(workflow)
    print(f"    Completed in {result.total_time_ms:.1f}ms")
    print(f"    Tasks completed: {len(result.results)}")
    if result.errors:
        print(f"    Errors: {result.errors}")

    # Show results
    print("\n[4] Results:")
    for task_id, response in result.results.items():
        print(f"    [{task_id}] {response[:120]}...")

    # Memory stats
    mem = fabric.memory_stats()
    print(f"\n[5] Memory: VRAM used={mem.vram_used_mb}MB free={mem.vram_free_mb}MB")

    fabric.shutdown()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
