#!/usr/bin/env python3
"""
Claude CUAD‑QA Output Generation Script
Uses prompt‑caching (5‑minute TTL) + batch API for cost‑efficient evaluation.
"""

import os, json, asyncio, argparse, hashlib, re
from datetime import datetime
from typing import List, Dict, Any, Optional

from datasets import load_dataset
from anthropic import Anthropic

class ClaudeCUADGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        max_seq_length: int = 65_536,
        batch_size: int = 1000,
    ):
        # 5‑minute cache; beta header kept (harmless)
        self.client = Anthropic(
            api_key=api_key,
            default_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"}
        )
        self.model = model
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.system_prompt = {
            "type": "text",
            "text": (
                """You are a legal document analyzer. Extract exact phrase from legal document to answer question. Only provide the exact text/phrase from the document that answer the question. Do not add explanations or commentary. If the information is not found, respond with 'Not found'."""
            ),
            "cache_control": {"type": "ephemeral"}  # 5‑minute TTL
        }
        self.id_map: Dict[str, str] = {}

    # ────────── prompt helpers ──────────
    def prepare_document_context(self, context: str) -> Dict[str, Any]:
        max_ctx = self.max_seq_length - 1000
        if len(context) > max_ctx:
            h = max_ctx // 2
            context = context[:h] + "\n[...TRUNCATED...]\n" + context[-h:]
        return {
            "type": "text",
            "text": f"Document: {context}",
            "cache_control": {"type": "ephemeral"}  # 5‑minute TTL
        }

    @staticmethod
    def prepare_prompt(question: str) -> str:
        return f"Question: {question}\n\nAnswer (extract exact phrase from document):"

    # ────────── grouping / priming ──────────
    @staticmethod
    def group_by_context(dataset) -> Dict[str, Any]:
        groups: Dict[str, Any] = {}
        for i, ex in enumerate(dataset):
            ctx = ex["context"]
            orig_id = ex.get("id", f"q_{i}")
            key = f"doc_{hash(ctx) % 1_000_000}"
            groups.setdefault(key, {"context": ctx, "questions": []})
            groups[key]["questions"].append({"id": orig_id, "question": ex["question"]})
        return groups

    def create_cache_prime_request(self, context: str) -> Dict[str, Any]:
        return {
            "custom_id": f"prime_{hash(context) % 1_000_000}",
            "params": {
                "model": self.model,
                "max_tokens": 5,
                "system": [self.system_prompt, self.prepare_document_context(context)],
                "messages": [{"role": "user", "content": "Ready?"}]
            }
        }

    async def prime_cache(self, contexts: List[str]) -> None:
        print(f"🔥 Priming cache for {len(contexts)} contexts…")
        primes = [self.create_cache_prime_request(c) for c in contexts]
        chunk = min(100, len(primes))
        for i in range(0, len(primes), chunk):
            batch = primes[i:i+chunk]
            bid = self.client.beta.messages.batches.create(requests=batch).id
            await self.wait_for_completion(bid, 5)
            print(f"✅ Prime batch {i//chunk+1} done")
        print("✅ Cache primed!")

    # ────────── batch builder ──────────
    def prepare_batch_requests(
        self,
        context_groups: Dict[str, Any],
        start_idx: int = 0
    ) -> List[Dict[str, Any]]:
        self.id_map.clear()
        reqs, idx = [], 0
        for grp in context_groups.values():
            doc_ctx = self.prepare_document_context(grp["context"])
            for q in grp["questions"]:
                if idx < start_idx:
                    idx += 1; continue
                if len(reqs) >= self.batch_size:
                    break

                orig = q["id"]
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", orig)
                if len(safe) > 64:
                    safe = f"{safe[:48]}_{hashlib.sha1(safe.encode()).hexdigest()[:15]}"
                self.id_map[safe] = orig

                reqs.append({
                    "custom_id": safe,
                    "params": {
                        "model": self.model,
                        "max_tokens": 64,
                        "system": [self.system_prompt, doc_ctx],
                        "messages": [{"role": "user",
                                      "content": self.prepare_prompt(q["question"])}]
                    }
                })
                idx += 1
            if len(reqs) >= self.batch_size:
                break
        return reqs

    # ────────── batch helpers ──────────
    async def submit_batch_job(self, requests: List[Dict[str, Any]], name="") -> Optional[str]:
        """Submit a batch and return its ID (sync call inside)."""
        try:
            batch = self.client.beta.messages.batches.create(requests=requests)
            print(f"🚀 {name} ID {batch.id}")
            return batch.id
        except Exception as e:
            print(f"❌ Submit error: {e}")
            return None

    async def wait_for_completion(self, batch_id: str, interval: int = 30):
        """Poll until batch ends/errs."""
        while True:
            info = self.client.beta.messages.batches.retrieve(batch_id)
            stat = info.processing_status
            rc = info.request_counts or None
            if rc:
                tot = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
                done = rc.succeeded + rc.errored + rc.canceled + rc.expired
                print(f"  {stat} {done}/{tot}")
            else:
                print(f"  {stat}")
            if stat == "ended" or stat in ("failed", "canceled"):
                return info
            await asyncio.sleep(interval)

    async def download_results(self, batch_id: str) -> Dict[str, str]:
        results = self.client.beta.messages.batches.results(batch_id)
        out: Dict[str, str] = {}
        for r in results:
            if r.result.type == "succeeded":
                out[self.id_map.get(r.custom_id, r.custom_id)] = (
                    r.result.message.content[0].text.strip()
                )
            else:
                out[self.id_map.get(r.custom_id, r.custom_id)] = f"Error: {repr(r.result.error)}"
                print(f"❌ {r.custom_id} → {repr(r.result.error)}")
        return out

    # ────────── top‑level run routine ──────────
    async def full_run(
        self,
        dataset_split: str,
        output_dir: str,
        max_samples: Optional[int] = None,
    ):
        ds = load_dataset("theatticusproject/cuad-qa", split=dataset_split, trust_remote_code=True)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        print(f"✅ Loaded {len(ds)} examples")

        groups = self.group_by_context(ds)
        print(f"✅ {len(groups)} unique document contexts")

        await self.prime_cache([g["context"] for g in groups.values()])

        all_preds: Dict[str, str] = {}
        total = sum(len(g["questions"]) for g in groups.values())
        print(f"📊 Processing {total} questions (batch size {self.batch_size})")

        start, num = 0, 1
        while start < total:
            reqs = self.prepare_batch_requests(groups, start)
            bid = await self.submit_batch_job(reqs, f"batch_{num}")
            if not bid:
                break
            await self.wait_for_completion(bid, 10)
            all_preds.update(await self.download_results(bid))
            start += len(reqs)
            num += 1

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"claude_preds_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(out_path, "w") as f:
            json.dump(all_preds, f, indent=2)
        print(f"📁 Saved predictions to {out_path}")

# ───────────── CLI / smoke test entry ─────────────
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api_key", required=True)
    p.add_argument("--split", default="test", choices=["train","test"])
    p.add_argument("--max_samples", type=int)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--output_dir", default="./claude_results")
    args = p.parse_args()

    gen = ClaudeCUADGenerator(api_key=args.api_key, batch_size=args.batch_size)

    # small smoke‑test path (≤10 examples) still calls submit_batch_job directly
    if args.max_samples and args.max_samples <= 10:
        ds = load_dataset("theatticusproject/cuad-qa", split=args.split, trust_remote_code=True).select(range(args.max_samples))
        groups = gen.group_by_context(ds)
        await gen.prime_cache([g["context"] for g in groups.values()])
        reqs = gen.prepare_batch_requests(groups, 0)
        bid = await gen.submit_batch_job(reqs, "smoke_test")
        if bid:
            await gen.wait_for_completion(bid, 5)
            preds = await gen.download_results(bid)
            print(json.dumps(preds, indent=2))
    else:
        await gen.full_run(args.split, args.output_dir, args.max_samples)

if __name__ == "__main__":
    asyncio.run(main())
