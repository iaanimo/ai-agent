"""
AI Agent Framework — Main Entry Point
=======================================
智能体框架主入口，支持交互式命令行。

Usage:
    python main.py                    # 交互式聊天
    python main.py --mode react       # ReAct 模式
    python main.py --mode plan        # Plan-and-Execute 模式
    python main.py --mode direct      # 直接回答模式
    python main.py --demo rag         # 运行 RAG 演示
    python main.py --demo cot         # 运行 CoT 演示
"""

import asyncio
import argparse
import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def print_banner():
    """Show the startup banner (auto-aligned, lists all commands)."""
    lines = [
        "🤖  AI Agent Framework",
        "",
        "Features:",
        "  • ReAct / Plan-and-Execute / Direct modes",
        "  • Chain of Thought (CoT) Reasoning",
        "  • RAG (Retrieval-Augmented Generation)",
        "  • Multi-Agent Collaboration",
        "  • Tool Calling & Function Calling",
        "",
        "Commands:",
        "  /mode <react|plan|direct>     Switch mode",
        "  /multi <task>                 Multi-agent (seq/para/debate)",
        "  /rag <path>                   Ingest documents",
        "  /think <problem>              Pure CoT reasoning",
        "  /remember <fact>              Remember a fact",
        "  /memories                     Show what it remembers",
        "  /forget <id|keyword>          Delete memories",
        "  /reset                        Reset memory",
        "  /quit                         Exit",
    ]
    console.print(Panel("\n".join(lines), title="AI Agent Framework", border_style="bold cyan"))


async def interactive_chat(mode: str = "react"):
    """Run interactive chat loop."""
    from core.agent import Agent, AgentMode
    from core.reasoning import CoTStrategy
    from tools import get_all_tools
    from rag import RAGRetriever
    from config.settings import get_settings
    from multi_agent import Orchestrator, ResearcherAgent, CoderAgent, ReviewerAgent
    from multi_agent.orchestrator import CollaborationMode

    print_banner()

    # Initialize agent
    agent_mode_map = {
        "react": AgentMode.REACT,
        "plan": AgentMode.PLAN_AND_EXECUTE,
        "direct": AgentMode.DIRECT,
    }

    agent = Agent(
        mode=agent_mode_map.get(mode, AgentMode.REACT),
        tools=get_all_tools(),
        enable_cot=True,
        cot_strategy=CoTStrategy.ZERO_SHOT,
        max_iterations=10,
        verbose=True,
        auto_memory=get_settings().agent.auto_memory,
    )

    rag = RAGRetriever(llm=agent.llm)
    agent.rag = rag  # Wire RAG into the agent so plan-mode 'rag_search' steps work

    console.print(f"\n✅ Agent ready in [bold green]{mode}[/bold green] mode")
    console.print("   Type your message or use /commands\n")

    while True:
        try:
            user_input = console.input("[bold blue]You> [/bold blue]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd == "/quit" or cmd == "/exit":
                    console.print("👋 Goodbye!", style="bold yellow")
                    break

                elif cmd == "/mode":
                    if len(parts) > 1:
                        new_mode = parts[1].strip().lower()
                        if new_mode in agent_mode_map:
                            agent.mode = agent_mode_map[new_mode]
                            console.print(f"✅ Switched to [bold]{new_mode}[/bold] mode")
                        elif new_mode == "multi":
                            console.print("ℹ️  Multi-agent is available as a command: /multi <任务> (e.g. /multi parallel 帮我设计一个推荐系统)")
                        else:
                            console.print(f"❌ Unknown mode: {new_mode}. Use: react, plan, direct")
                    else:
                        console.print(f"Current mode: [bold]{agent.mode.value}[/bold]")

                elif cmd == "/rag":
                    if len(parts) > 1:
                        path = parts[1].strip()
                        try:
                            count = rag.ingest(path)
                            console.print(f"✅ Ingested {count} chunks from {path}")
                        except Exception as e:
                            console.print(f"❌ Error: {e}", style="red")
                    else:
                        console.print("Usage: /rag <file_or_directory_path>")

                elif cmd == "/multi":
                    rest = parts[1].strip() if len(parts) > 1 else ""
                    if not rest:
                        console.print("Usage: /multi [sequential|parallel|debate] <任务>")
                        continue
                    words = rest.split(maxsplit=1)
                    mode_map = {
                        "sequential": CollaborationMode.SEQUENTIAL,
                        "parallel": CollaborationMode.PARALLEL,
                        "debate": CollaborationMode.DEBATE,
                    }
                    mode_word = words[0].lower()
                    if mode_word in mode_map:
                        task = words[1].strip() if len(words) > 1 else ""
                        mode = mode_map[mode_word]
                    else:
                        task = rest
                        mode = CollaborationMode.SEQUENTIAL
                    if not task:
                        console.print("Usage: /multi [sequential|parallel|debate] <任务>")
                        continue

                    console.print(f"🤝 多智能体协作开始（{mode.value} 模式）...")
                    orchestrator = Orchestrator(llm=agent.llm, verbose=True)
                    orchestrator.add_agent(ResearcherAgent(llm=agent.llm))
                    orchestrator.add_agent(CoderAgent(llm=agent.llm))
                    orchestrator.add_agent(ReviewerAgent(llm=agent.llm))
                    result = await orchestrator.run(task, mode=mode)
                    console.print()
                    console.print(Panel(Markdown(result.final_answer), title="🤝 多智能体协作结果", border_style="magenta"))

                    # Feed the multi-agent work back into the main agent's memory,
                    # so a later "你帮我设计过 X 吗 / 我要怎么使用" can recall it.
                    if agent.memory:
                        agent.memory.add_user_message(f"[多智能体任务] {task}")
                        agent.memory.add_ai_message(f"[多智能体协作结果]\n{result.final_answer}")
                        agent.memory.store_fact(
                            f"为用户完成了多智能体任务「{task[:60]}」，结论摘要：{result.final_answer[:300]}",
                            category="task_output",
                        )

                elif cmd == "/think":
                    if len(parts) > 1:
                        problem = parts[1].strip()
                        chain = await agent.think(problem)
                        console.print(Panel(chain.to_display(), title="Chain of Thought", border_style="green"))

                        # Same: remember the reasoning result for later recall.
                        if agent.memory:
                            agent.memory.add_user_message(f"[思维链任务] {problem}")
                            agent.memory.add_ai_message(f"[思维链结论]\n{chain.conclusion}")
                    else:
                        console.print("Usage: /think <problem>")

                elif cmd == "/reset":
                    agent.reset()
                    console.print("🔄 Memory reset")

                elif cmd == "/remember":
                    if len(parts) > 1:
                        if agent.memory and agent.memory.store_fact(parts[1].strip()):
                            console.print("🧠 Remembered.")
                        else:
                            console.print("ℹ️  Already knows that (skipped as duplicate).")
                    else:
                        console.print("Usage: /remember <fact>")

                elif cmd == "/memories":
                    if agent.memory and agent.memory.long_term:
                        entries = agent.memory.long_term.get_all()
                        if not entries:
                            console.print("No memories stored yet. They accumulate automatically as you talk.")
                        else:
                            for e in entries:
                                console.print(f"  [{e.get('id', '?')}] ({e.get('category', 'general')}) "
                                              f"{e['content']}  🔁{e.get('access_count', 0)}")
                    else:
                        console.print("Memory is disabled.")

                elif cmd == "/forget":
                    if len(parts) > 1:
                        if agent.memory and agent.memory.long_term:
                            removed = agent.memory.forget(parts[1].strip())
                            console.print(f"🗑️ Removed {removed} memory/memories."
                                          if removed else "No matching memory found.")
                        else:
                            console.print("Memory is disabled.")
                    else:
                        console.print("Usage: /forget <memory_id_or_keyword>")

                else:
                    console.print(f"❌ Unknown command: {cmd}")

                continue

            # Regular chat
            console.print()
            result = await agent.run(user_input)
            console.print(Panel(Markdown(result), title="🤖 Agent", border_style="green"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {e}", style="red")


async def run_brief():
    """Generate today's AI-industry briefing and save it under data/.

    A "daily routine" for the JARVIS-style assistant: reuse the same agent
    pipeline that produced ai_brief.md by hand in chat — search → synthesize →
    final_answer (writes the file). If the agent forgets to attach a file_path,
    we save the returned markdown ourselves so the briefing is always on disk.
    """
    from core.agent import Agent, AgentMode
    from tools.search import create_search_tool
    from tools.file_ops import resolve_data_path

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"ai_brief_{today}.md"
    target = resolve_data_path(filename)

    task = (
        f"请帮我生成今天（{today}）的《人工智能行业早报》。\n"
        f"步骤：1) 只用 search 工具搜索今天最新的 AI 行业新闻（模型发布、公司动态、融资、政策监管），最多搜索 3 次；\n"
        f"注意：只使用 search 工具，不要调用 code_executor 抓网页，也不要使用任何其他工具。\n"
        f"2) 无论搜索到多少素材，最多 3 次搜索后必须调用 final_answer；挑 3-5 条最有价值的，"
        f"按板块整理成简洁的 Markdown 早报（标题 + 分条，注明每条新闻的时间）；\n"
        f"3) 如果确实搜不到今天的新闻，就如实说明，不要拿旧闻冒充今日；\n"
        f"4) 用 final_answer 工具给出完整 Markdown 内容，并在 args 里加上 "
        f'"file_path": "{filename}"（不要加 data/ 前缀）把早报保存成文件。'
    )

    # 早报只需"搜索 → 整理"，只给 search 一个工具，防止 agent 跑偏去抓网页。
    agent = Agent(
        mode=AgentMode.REACT,
        tools=[create_search_tool()],
        enable_memory=False,   # daily routine: 不往长期记忆里记"用户事实"
        enable_cot=False,
        max_iterations=8,
        verbose=True,
        auto_memory=False,
    )

    console.print(f"📰 正在生成今日（{today}）AI 行业早报...")
    result = await agent.run(task)

    # 双保险：agent 若已通过 final_answer 落盘，这里不会重复写。
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result, encoding="utf-8")
        console.print(f"📄 早报已保存到 {target}", style="green")

    console.print()
    console.print(Panel(Markdown(result), title="📰 AI 行业早报", border_style="green"))


async def run_demo(demo_name: str):
    """Run a specific demo."""
    if demo_name == "rag":
        from examples.rag_demo import main as rag_main
        await rag_main()
    elif demo_name == "cot":
        from examples.cot_demo import main as cot_main
        await cot_main()
    elif demo_name == "multi":
        from examples.multi_agent_demo import main as multi_main
        await multi_main()
    elif demo_name == "plan":
        from examples.plan_and_execute_demo import main as plan_main
        await plan_main()
    elif demo_name == "single":
        from examples.single_agent_demo import main as single_main
        await single_main()
    else:
        console.print(f"❌ Unknown demo: {demo_name}")
        console.print("Available demos: rag, cot, multi, plan, single")


def main():
    # Windows 中文环境下终端常为 GBK 编码，rich 输出的 emoji / 中文可能触发
    # UnicodeEncodeError。统一切到 UTF-8 输出（含错误替换兜底），保证任何
    # 模式下面板和 emoji 都能正常显示。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(description="AI Agent Framework")
    parser.add_argument("--mode", default="react", choices=["react", "plan", "direct"],
                        help="Agent running mode")
    parser.add_argument("--demo", type=str, help="Run a demo (rag, cot, multi, plan, single)")
    parser.add_argument("--brief", action="store_true",
                        help="Generate today's AI briefing and save it under data/")

    args = parser.parse_args()

    if args.brief:
        asyncio.run(run_brief())
    elif args.demo:
        asyncio.run(run_demo(args.demo))
    else:
        asyncio.run(interactive_chat(args.mode))


if __name__ == "__main__":
    main()
