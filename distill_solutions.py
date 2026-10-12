#!/usr/bin/env python3
"""
Generate solutions for problems in a JSONL dataset using an OpenAI-compatible API
(local vLLM server or remote API endpoint).

Supports concurrent workers, resume from checkpoint, and incremental writes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from openai import OpenAI
import threading
from queue import Queue
from tqdm import tqdm
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


class SolutionDistiller:
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        data_path: str,
        save_path: str,
        model_name: str = None,
        tokenizer_path: str = None,
        n_samples: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 8192,
        num_threads: int = 10,
        resume: bool = True,
        include_sources: str = None,
        enable_thinking: Optional[bool] = None,
        retry_until_complete: bool = False,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.model_name = model_name or model
        self.tokenizer_path = tokenizer_path or model
        self.data_path = data_path
        self.save_path = save_path
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.resume = resume
        self.include_sources = set(include_sources.split(',')) if include_sources else None
        self.enable_thinking = enable_thinking
        self.retry_until_complete = retry_until_complete
        # Explicit 1200 s timeout (double the httpx/OpenAI default of 600 s)
        # to accommodate long-running reasoning model responses.
        self.request_timeout = 1200.0

        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=self.request_timeout)

        self.tokenizer = None
        if self.enable_thinking is not None:
            if AutoTokenizer is None:
                print("WARNING: transformers not found; cannot use apply_chat_template, falling back to plain prompt")
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
                    print(
                        f"Tokenizer loaded: {self.tokenizer_path} "
                        f"(enable_thinking={self.enable_thinking})"
                    )
                except Exception as e:
                    print(f"WARNING: tokenizer load failed, falling back to plain prompt: {e}")

        self.write_lock = threading.Lock()

        self.started_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.initial_completed_count = 0
        self.lock = threading.Lock()

    def load_data(self) -> List[Dict]:
        """Load input dataset."""
        print(f"Loading data: {self.data_path}")
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if self.include_sources is None or item.get('source') in self.include_sources:
                    data.append(item)
        print(f"Loaded {len(data)} items")
        if self.include_sources:
            print(f"Source filter: {', '.join(self.include_sources)}")
        return data

    def get_completed_items(self) -> Set[Tuple[str, int, int]]:
        """Return (source, index, sample) triples already written to disk."""
        if not self.resume:
            print("Restart mode: overwriting existing output")
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
                print(f"Removed existing file: {self.save_path}")
            return set()
        return self.scan_completed_items(print_status=True)

    def scan_completed_items(self, print_status: bool = False) -> Set[Tuple[str, int, int]]:
        """Scan the output file for completed (source, index, sample) triples."""
        completed = set()
        if os.path.exists(self.save_path):
            if print_status:
                print(f"Resume mode: scanning completed items in {self.save_path}")
            with open(self.save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if self.include_sources is None or item.get('source') in self.include_sources:
                            completed.add((item['source'], item['index'], item['sample']))
                    except Exception:
                        continue
            if print_status:
                print(f"Found {len(completed)} completed items; resuming from checkpoint")
        elif print_status:
            print("No existing output found; starting from scratch")
        return completed

    def build_pending_tasks(self, data: List[Dict], completed: Set[Tuple[str, int, int]]) -> List[Tuple[Dict, int]]:
        """Build the list of (item, sample_id) tasks that are not yet completed."""
        tasks = []
        for item in data:
            for sample_id in range(self.n_samples):
                if (item['source'], item['index'], sample_id) not in completed:
                    tasks.append((item, sample_id))
        return tasks

    def reset_round_counters(self):
        self.started_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.total_count = 0

    def generate_solution(self, problem: str, prompt_override: Optional[str] = None) -> str:
        """Call the API to generate one solution.

        If the item contains a 'prompt' field (code/SWE tasks), use it directly as
        the user message instead of the default math \\boxed{} wrapper.
        """
        if prompt_override is not None:
            user_prompt = prompt_override
        else:
            user_prompt = (
                f"Problem: {problem}\n\n"
                "Please solve this problem step by step and put your final answer "
                "in the format \\boxed{}."
            )

        prompt = None
        if self.tokenizer is not None and self.enable_thinking is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except Exception as e:
                print(f"WARNING: apply_chat_template failed, falling back to plain prompt: {e}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if prompt is not None:
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )
                    return response.choices[0].text
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {wait_time}s ...")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return ""

    def write_result(self, result: Dict):
        """Thread-safe append of a single result to the output file."""
        with self.write_lock:
            with open(self.save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def process_item(self, item: Dict, sample_id: int, completed: Set[Tuple[str, int, int]]) -> bool:
        """Generate and save one sample for one item."""
        source = item['source']
        index = item['index']

        if (source, index, sample_id) in completed:
            return False

        generated_solution = self.generate_solution(
            item['problem'],
            prompt_override=item.get('prompt'),
        )

        if not generated_solution:
            print(f"WARNING: empty generation for index={index}, sample={sample_id}")
            with self.lock:
                self.failed_count += 1
            return False

        result = {
            'problem': item['problem'],
            'original_solution': item.get('original_solution', ''),
            'answer': item['answer'],
            'source': item['source'],
            'index': index,
            'generated_solution': generated_solution,
            'sample': sample_id,
            'model': self.model_name,
        }

        self.write_result(result)

        with self.lock:
            self.completed_count += 1

        return True

    def worker(self, task_queue: Queue, completed: Set[Tuple[str, int, int]]):
        """Worker thread: drain the task queue."""
        while True:
            task = task_queue.get()
            if task is None:
                break
            item, sample_id = task
            with self.lock:
                self.started_count += 1
            self.process_item(item, sample_id, completed)
            task_queue.task_done()

    def run_round(
        self,
        tasks: List[Tuple[Dict, int]],
        completed: Set[Tuple[str, int, int]],
        total_target: int,
        round_idx: int,
    ):
        """Execute one round of tasks and return summary statistics."""
        self.reset_round_counters()
        self.total_count = len(tasks)

        print(f"\n--- Round {round_idx} ---")
        print(f"Total target : {total_target}")
        print(f"Already done : {len(completed)}")
        print(f"Pending      : {self.total_count}")

        if self.total_count == 0:
            print("All tasks already complete.")
            return {"elapsed": 0.0, "started": 0, "success": 0, "failed": 0}

        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)

        print(f"\nStarting {self.num_threads} worker threads ...")
        threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.worker, args=(task_queue, completed))
            t.start()
            threads.append(t)

        start_time = time.time()
        is_tty = sys.stdout.isatty() and sys.stderr.isatty()
        log_interval = 10.0
        last_log_t = start_time

        print("┌──────────┬─────────────┬─────────────┬────────┬────────┬────────────┬────────────┐")
        print("│ Elapsed  │ Sent/Total  │ In-flight   │ Done   │ Failed │ Rate       │ ETA        │")
        print("├──────────┼─────────────┼─────────────┼────────┼────────┼────────────┼────────────┤")

        while task_queue.unfinished_tasks > 0:
            time.sleep(2)
            with self.lock:
                elapsed = time.time() - start_time
                processed = self.completed_count + self.failed_count
                in_flight = max(self.started_count - processed, 0)
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (self.total_count - processed) / rate if rate > 0 else float("inf")
                eta_str = f"{eta/60:.1f} min" if eta != float("inf") else "unknown"
                rate_str = f"{rate:.3f}/s" if rate < 0.1 else f"{rate:.2f}/s"
                if is_tty:
                    pct = self.started_count / self.total_count * 100 if self.total_count else 0
                    print(
                        f"\r  {elapsed/60:.1f} min | {self.started_count}/{self.total_count} "
                        f"({pct:.1f}%) | done={self.completed_count} fail={self.failed_count} "
                        f"rate={rate_str} eta={eta_str}",
                        end='', flush=True,
                    )
                now = time.time()
                if now - last_log_t >= log_interval:
                    print(
                        f"\n│ {elapsed/60:6.1f} min │ {self.started_count}/{self.total_count:<7} │ "
                        f"{in_flight:<11} │ {self.completed_count:<6} │ {self.failed_count:<6} │ "
                        f"{rate_str:<10} │ {eta_str:<10} │",
                        flush=True,
                    )
                    last_log_t = now

        task_queue.join()
        for _ in range(self.num_threads):
            task_queue.put(None)
        for t in threads:
            t.join()

        elapsed = time.time() - start_time
        if is_tty:
            print()
        print("└──────────┴─────────────┴─────────────┴────────┴────────┴────────────┴────────────┘")
        print(f"\nRound {round_idx} complete | elapsed={elapsed/60:.1f} min | "
              f"sent={self.started_count} done={self.completed_count} failed={self.failed_count}")
        avg = (self.completed_count + self.failed_count) / elapsed if elapsed > 0 else 0
        avg_str = f"{avg:.3f}/s" if avg < 0.1 else f"{avg:.2f}/s"
        print(f"Average throughput: {avg_str}")
        print(f"Output: {self.save_path}")
        return {
            "elapsed": elapsed,
            "started": self.started_count,
            "success": self.completed_count,
            "failed": self.failed_count,
        }

    def run(self):
        """Main entry point."""
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)

        data = self.load_data()
        completed = self.get_completed_items()
        self.initial_completed_count = len(completed)
        total_target = len(data) * self.n_samples

        overall = {"started": 0, "success": 0, "failed": 0, "elapsed": 0.0}
        round_idx = 1

        while True:
            if round_idx > 1:
                completed = self.scan_completed_items(print_status=False)

            tasks = self.build_pending_tasks(data, completed)
            stats = self.run_round(tasks, completed, total_target, round_idx)

            for k in ("started", "success", "failed", "elapsed"):
                overall[k] += stats[k]

            completed_after = self.scan_completed_items(print_status=False)
            if len(completed_after) >= total_target:
                print(f"\nAll {len(completed_after)}/{total_target} samples complete.")
                break

            if not self.retry_until_complete:
                break

            remaining = total_target - len(completed_after)
            print(f"\nAuto-retry enabled: {remaining} samples still missing, starting next round ...")
            if stats["success"] == 0:
                print("WARNING: no new successes this round; retrying anyway.")
            round_idx += 1

        if self.retry_until_complete:
            print("\n" + "=" * 60)
            print("Auto-retry summary")
            print("=" * 60)
            print(f"  Rounds          : {round_idx}")
            print(f"  Initial done    : {self.initial_completed_count}")
            print(f"  Total sent      : {overall['started']}")
            print(f"  Total success   : {overall['success']}")
            print(f"  Total failed    : {overall['failed']}")
            avg = (overall['success'] + overall['failed']) / overall['elapsed'] if overall['elapsed'] > 0 else 0
            avg_str = f"{avg:.3f}/s" if avg < 0.1 else f"{avg:.2f}/s"
            print(f"  Avg throughput  : {avg_str}")
            print(f"  Output          : {self.save_path}")


def main():
    def str2bool(value: str) -> bool:
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")

    parser = argparse.ArgumentParser(description="Generate solutions using an OpenAI-compatible API")

    # API
    parser.add_argument('--api_key',         type=str, default='EMPTY', help='API key (default: EMPTY for local vLLM)')
    parser.add_argument('--api_base',        type=str, required=True,   help='API base URL')
    parser.add_argument('--model',           type=str, required=True,   help='Model name sent in each API request')
    parser.add_argument('--tokenizer_path',  type=str, default=None,
                        help='Path to tokenizer (used for apply_chat_template when enable_thinking is set)')

    # Data
    parser.add_argument('--data_path',   type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--save_path',   type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--model_name',  type=str, default=None,  help='Model name written into output records')

    # Generation
    parser.add_argument('--n_samples',   type=int,   default=8,     help='Samples per problem')
    parser.add_argument('--temperature', type=float, default=0.7,   help='Sampling temperature')
    parser.add_argument('--top_p',       type=float, default=0.95,  help='Top-p nucleus sampling')
    parser.add_argument('--max_tokens',  type=int,   default=8192,  help='Max tokens per response')

    # Concurrency
    parser.add_argument('--num_threads', type=int, default=10, help='Number of concurrent worker threads')

    # Resume / retry
    parser.add_argument('--no_resume',           action='store_true',
                        help='Disable resume; overwrite existing output')
    parser.add_argument('--retry_until_complete', action='store_true',
                        help='Keep retrying failed items until every problem has n_samples results')

    # Filters
    parser.add_argument('--include_sources', type=str, default=None,
                        help='Comma-separated list of sources to include (default: all)')
    parser.add_argument('--enable_thinking', type=str2bool, default=None,
                        help='Thinking toggle (true/false); triggers apply_chat_template when set')

    args = parser.parse_args()

    print("=" * 60)
    print("Solution Distiller")
    print("=" * 60)
    print(f"  model          : {args.model}")
    print(f"  tokenizer      : {args.tokenizer_path or '(not set)'}")
    print(f"  api_base       : {args.api_base}")
    print(f"  data_path      : {args.data_path}")
    print(f"  save_path      : {args.save_path}")
    print(f"  n_samples      : {args.n_samples}")
    print(f"  temperature    : {args.temperature}")
    print(f"  top_p          : {args.top_p}")
    print(f"  max_tokens     : {args.max_tokens}")
    print(f"  num_threads    : {args.num_threads}")
    print(f"  sources        : {args.include_sources or 'all'}")
    print(f"  enable_thinking: {args.enable_thinking if args.enable_thinking is not None else '(not set)'}")
    print(f"  resume         : {'no (restart)' if args.no_resume else 'yes'}")
    print(f"  auto-retry     : {'yes' if args.retry_until_complete else 'no'}")
    print("=" * 60)

    distiller = SolutionDistiller(
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        save_path=args.save_path,
        model_name=args.model_name,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_threads=args.num_threads,
        resume=not args.no_resume,
        include_sources=args.include_sources,
        enable_thinking=args.enable_thinking,
        retry_until_complete=args.retry_until_complete,
    )

    distiller.run()


if __name__ == "__main__":
    main()
