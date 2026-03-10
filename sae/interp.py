"""SAE 特征解释: 收集激活上下文 + LLM API 自动评分"""

import heapq, json, os, random, re, time, logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .sae_model import TopKSAE

logger = logging.getLogger(__name__)

SENTENCE_ENDERS = {".", "!", "?", "<|end_of_text|>", '"'}
_RESP_RE = re.compile(
    r"(?:\*)*\s*Score\s*(?:\*)*\s*[:\-]?\s*(?:\*)*\s*(?P<score>[1-5])\s*(?:\*)*\s*"
    r"(?:\r?\n|\s)+"
    r"(?:\*)*\s*Explanation\s*(?:\*)*\s*[:\-]?\s*(?:\*)*\s*(?P<explanation>.+)",
    re.IGNORECASE | re.DOTALL,
)


def _tokens_to_str(tokens: List[str]) -> str:
    """token 列表 → 可读字符串（处理 sentencepiece/BPE 前缀）"""
    parts = []
    for t in tokens:
        if t.startswith("▁") or t.startswith("Ġ"):
            parts.append(" " + t[1:])
        else:
            parts.append(t)
    return "".join(parts).strip()


def _build_context(seq_pos: int, tokens: List[str], max_len: int = 64):
    """提取 token 所在句子上下文，标记激活位置。返回 (text, raw_token) 或 None"""
    # 句子边界
    s, e = seq_pos, seq_pos
    while s > 0 and tokens[s - 1] not in SENTENCE_ENDERS:
        s -= 1
    while e < len(tokens) - 1 and tokens[e] not in SENTENCE_ENDERS:
        e += 1
    if e < len(tokens):
        e += 1

    ctx = tokens[s:e]
    ai = seq_pos - s  # activated index

    # 截断过长上下文
    if len(ctx) > max_len:
        half = max_len // 2
        lo = max(0, ai - half)
        hi = min(len(ctx), ai + half + 1)
        ctx, ai = ctx[lo:hi], ai - lo

    if not (0 <= ai < len(ctx)):
        return None

    ctx = list(ctx)
    raw = ctx[ai]
    ctx[ai] = f"<ACTIVATED>{raw}</ACTIVATED>"

    # 清理首尾
    while ctx and ctx[0] in ("<|end_of_text|>", " ", ""):
        ctx.pop(0); ai -= 1
    while ctx and ctx[-1] in ("<|end_of_text|>", " ", ""):
        ctx.pop()
    if not ctx or not (0 <= ai < len(ctx)):
        return None

    text = _tokens_to_str(ctx).strip().strip('"')
    return (text, raw.strip()) if text else None


def _add_selected_prefix(path: str) -> str:
    """给输出文件名添加 selected_ 前缀。"""
    directory, filename = os.path.split(path)
    if filename.startswith("selected_"):
        return path
    return os.path.join(directory, f"selected_{filename}")


# ── ContextCollector ─────────────────────────────────────────

class ContextCollector:
    """收集 SAE 每个特征的高激活上下文片段。"""

    def __init__(self, model: nn.Module, tokenizer, sae: TopKSAE,
                 layer: int, sae_path: str = "", device: str = "cuda"):
        self.model, self.tokenizer, self.sae = model, tokenizer, sae
        self.layer, self.device = layer, device
        self.sae_name = os.path.splitext(os.path.basename(sae_path))[0] if sae_path else "sae"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._hidden: Optional[torch.Tensor] = None
        self._hook = None

    # -- hook --
    def _get_layer_module(self) -> nn.Module:
        for path in ("model.layers", "language_model.model.layers"):
            try:
                m = self.model
                for a in path.split("."):
                    m = getattr(m, a)
                return m[self.layer]
            except (AttributeError, IndexError):
                continue
        raise AttributeError(f"无法定位第 {self.layer} 层")

    def _register_hook(self):
        self._remove_hook()
        def fn(mod, inp, out):
            self._hidden = (out[0] if isinstance(out, tuple) else out).detach()
        self._hook = self._get_layer_module().register_forward_hook(fn)

    def _remove_hook(self):
        if self._hook is not None:
            self._hook.remove(); self._hook = None

    # -- 主逻辑 --
    @torch.no_grad()
    def collect(self, texts: Iterator[str], output_path: Optional[str] = None,
                threshold: float = 10.0, max_length: int = 64,
                max_per_token: int = 3, min_contexts: int = 5,
                max_token_classes: int = 0, max_seq_length: int = 1024,
                batch_size: int = 16,
                selected_feature_ids: Optional[List[int]] = None) -> Tuple[int, str]:
        """收集特征激活上下文 → JSON"""
        selected_set = set(selected_feature_ids) if selected_feature_ids else None
        if output_path is None:
            output_path = f"outputs/contexts/{self.sae_name}_{threshold}.json"
        if selected_set:
            output_path = _add_selected_prefix(output_path)
        self._register_hook()
        self.model.eval(); self.sae.eval()
        ctx_map: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        buf, n = [], 0

        for text in texts:
            buf.append(text)
            if len(buf) >= batch_size:
                self._process(buf, ctx_map, threshold, max_length, max_per_token, max_seq_length, selected_set)
                n += len(buf); buf = []
                if n % (batch_size * 10) == 0:
                    logger.info(f"已处理 {n} 条文本")
        if buf:
            self._process(buf, ctx_map, threshold, max_length, max_per_token, max_seq_length, selected_set)
            n += len(buf)

        self._remove_hook()
        logger.info(f"共处理 {n} 条文本, 原始特征数: {len(ctx_map)}")

        # 过滤 & 排序
        filtered = {}
        skip_classes, skip_contexts = 0, 0
        for dim, td in ctx_map.items():
            if max_token_classes > 0 and len(td) > max_token_classes:
                skip_classes += 1
                continue
            if sum(len(h) for h in td.values()) < min_contexts:
                skip_contexts += 1
                continue
            filtered[dim] = {
                tc: [{"context": c, "activation": round(a, 4)}
                     for a, c in sorted(heap, reverse=True)]
                for tc, heap in sorted(td.items())
            }
        filtered = dict(sorted(filtered.items()))
        logger.info(f"过滤: token类别过多={skip_classes}, 上下文不足={skip_contexts}, 保留={len(filtered)}")

        out = {"total_features": len(filtered), "threshold": threshold,
               "n_texts": n, "latent_context_map": {str(k): v for k, v in filtered.items()}}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"提取 {len(filtered)} 个特征上下文 → {output_path}")
        return len(filtered), output_path

    def _process(self, texts, ctx_map, threshold, max_length, max_per_token, max_seq_len, selected_set=None):
        inp = self.tokenizer(texts, return_tensors="pt", max_length=max_seq_len,
                             padding=True, truncation=True)
        ids = inp["input_ids"].to(self.device)
        self.model(input_ids=ids, attention_mask=inp["attention_mask"].to(self.device))
        if self._hidden is None:
            return
        h = self._hidden
        bs, sl, hd = h.shape
        latents = self.sae.encode(h.reshape(-1, hd).to(self.sae.config.get_torch_dtype()))
        latents = latents.reshape(bs, sl, -1)

        for i in range(bs):
            toks = self.tokenizer.convert_ids_to_tokens(ids[i].cpu().tolist())
            for pos, dim in torch.nonzero(latents[i] > threshold, as_tuple=False).tolist():
                if selected_set is not None and dim not in selected_set:
                    continue
                r = _build_context(pos, toks, max_length)
                if r is None:
                    continue
                ctx_text, raw = r
                heap = ctx_map[dim][raw.lower()]
                heapq.heappush(heap, (latents[i, pos, dim].item(), ctx_text))
                if len(heap) > max_per_token:
                    heapq.heappop(heap)


# ── FeatureInterpreter ───────────────────────────────────────
class FeatureInterpreter:
    """调用 LLM API 对 SAE 特征进行可解释性评分。"""

    SYSTEM_PROMPT = "You are an assistant that helps explain the latent semantics of language models."
    PROMPT_TEMPLATE = (
        "We are analyzing the activation levels of features in a neural network, "
        "where each feature activates certain tokens in a text.\n"
        "Each token's activation value indicates its relevance to the feature, "
        "with higher values showing stronger association.\n"
        "Your task is to give this feature a monosemanticity score based on the following scoring rubric:\n"
        "Activation Consistency\n"
        "5: Clear pattern with no deviating examples\n"
        "4: Clear pattern with one or two deviating examples\n"
        "3: Clear overall pattern but quite a few examples not fitting that pattern\n"
        "2: Broad consistent theme but lacking structure\n"
        "1: No discernible pattern\n"
        "Consider the following activations for a feature in the neural network.\n\n"
    )

    def __init__(self, api_key: str, model: str = "gpt-4o",
                 api_base: Optional[str] = None):
        self.model_name = model
        from openai import OpenAI
        self.client = OpenAI(
            base_url=api_base if api_base else "https://api.openai.com/v1",
            api_key=api_key
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value")
                else:
                    text = getattr(item, "text", None) or getattr(item, "content", None) or getattr(item, "value", None)
                text = FeatureInterpreter._content_to_text(text)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            return FeatureInterpreter._content_to_text(
                content.get("text") or content.get("content") or content.get("value")
            )
        return str(content).strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        text = FeatureInterpreter._content_to_text(getattr(message, "content", None))
        return FeatureInterpreter._strip_code_fences(text)

    def _chat(self, prompt: str, retries: int = 3) -> str:
        for i in range(1, retries + 1):
            try:
                try:
                    r = self.client.chat.completions.create(
                        messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                        model=self.model_name, max_completion_tokens=256, temperature=0.1)
                except TypeError:
                    r = self.client.chat.completions.create(
                        messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                        model=self.model_name, max_tokens=256, temperature=0.1)
                except Exception as e:
                    if "max_completion_tokens" not in str(e):
                        raise
                    r = self.client.chat.completions.create(
                        messages=[{"role": "system", "content": self.SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt}],
                        model=self.model_name, max_tokens=256, temperature=0.1)

                text = self._extract_chat_text(r)
                if not text:
                    raise ValueError("API returned empty content")
                return text.strip()
            except Exception as e:
                logger.warning(f"API 失败 ({i}/{retries}): {e}")
                if i < retries:
                    time.sleep(2 ** i)
                else:
                    raise

    @staticmethod
    def _build_prompt(tokens_info: List[Dict[str, Any]]) -> str:
        p = FeatureInterpreter.PROMPT_TEMPLATE
        for t in tokens_info:
            p += f"Token: {t['token']} | Activation: {t['activation']} | Context: {t['context']}\n\n"
        p += "Provide your response in the following fixed format:\nScore: [5/4/3/2/1]\nExplanation: [Your brief explanation]\n"
        return p

    def run(self, context_path: str, output_path: Optional[str] = None,
            sample_features: int = 100, seed: int = 42) -> Dict[str, Any]:
        """对采样特征执行 LLM 评分，返回 {avg_score, features_scored, model}"""
        if output_path is None:
            ctx_name = os.path.splitext(os.path.basename(context_path))[0]
            output_path = f"outputs/contexts/interp_{ctx_name}.json"
        random.seed(seed)
        with open(context_path, "r", encoding="utf-8") as f:
            ctx_map = json.load(f).get("latent_context_map", {})
        if sample_features >= len(ctx_map):
            keys = sorted(ctx_map.keys(), key=int)
        else:
            keys = sorted(random.sample(list(ctx_map.keys()), sample_features), key=int)
        logger.info(f"采样 {len(keys)}/{len(ctx_map)} 个特征")

        results, total, scored = {}, 0.0, 0
        for idx, key in enumerate(keys, 1):
            infos = []
            for tc, ctxs in ctx_map[key].items():
                tok = (" " + tc[1:]) if tc.startswith("ġ") or tc.startswith("▁") else tc
                for c in ctxs:
                    infos.append({"token": tok, "context": c["context"], "activation": c["activation"]})
            try:
                resp = self._chat(self._build_prompt(infos))
                m = _RESP_RE.search(resp)
                if m and 1 <= int(m.group("score")) <= 5:
                    s = int(m.group("score"))
                    results[key] = {"score": s, "explanation": m.group("explanation").strip()}
                    total += s; scored += 1
                else:
                    snippet = resp.replace("\n", " ")[:200]
                    results[key] = {"score": None, "explanation": f"Parse failed: {snippet}"}
            except Exception as e:
                logger.error(f"特征 {key}: {e}")
                results[key] = {"score": None, "explanation": str(e)}
            if idx % 10 == 0:
                logger.info(f"已解释 {idx}/{len(keys)}")

        avg = total / scored if scored else 0.0
        summary = {"avg_score": round(avg, 4), "features_scored": scored, "model": self.model_name}
        logger.info(f"完成: {scored} 个特征, 平均 {avg:.2f}")

        sorted_results = dict(sorted(results.items(), key=lambda kv: int(kv[0])))
        out = {**summary, "results": sorted_results}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return summary


# ── main ─────────────────────────────────────────────────────

def main():
    """使用范例：收集上下文 + LLM 解释（可直接 python -m sae.interp 运行）"""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="SAE 特征解释工具")
    sub = parser.add_subparsers(dest="cmd")

    # ---- collect 子命令 ----
    p1 = sub.add_parser("collect", help="收集特征激活上下文")
    p1.add_argument("--model_path", required=True, help="LLM 路径")
    p1.add_argument("--sae_path", required=True, help="SAE checkpoint 路径")
    p1.add_argument("--data_path", default="/data/agent_tool_use/Agent-Tool-Use-MI/data/raw/pretrain", help="JSONL 文件或文件夹路径")
    p1.add_argument("--target_tokens", type=int, default=10_000_000, help="目标 token 数 (默认 10M)")
    p1.add_argument("--layer", type=int, required=True, help="目标层 (0-indexed)")
    p1.add_argument("--output", default=None)
    p1.add_argument("--threshold", type=float, default=10.0)
    p1.add_argument("--batch_size", type=int, default=16)
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--feature_ids", type=int, nargs="+", default=None,
                    help="仅收集这些 feature id 的上下文")

    # ---- interpret 子命令 ----
    p2 = sub.add_parser("interpret", help="调用 LLM API 解释特征")
    p2.add_argument("--context_path", required=True, help="collect 输出的 JSON")
    p2.add_argument("--output", default=None)
    p2.add_argument("--api_key", required=True)
    p2.add_argument("--api_base", default=None)
    p2.add_argument("--llm_model", default="gpt-5")
    p2.add_argument("--sample", type=int, default=100)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "collect":
        # 加载 LLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        ).to(args.device).eval()
        sae = TopKSAE.load(args.sae_path, device=args.device)

        # 读取文本（支持单文件或文件夹，按 target_tokens 截断）
        import glob, jsonlines
        def read_texts():
            files = (sorted(glob.glob(os.path.join(args.data_path, "*.jsonl")))
                     if os.path.isdir(args.data_path) else [args.data_path])
            total = 0
            for fp in files:
                with jsonlines.open(fp) as reader:
                    for obj in reader:
                        text = obj.get("text", "")
                        total += len(text.split())
                        yield text
                        if total >= args.target_tokens:
                            logger.info(f"已达目标 token 数 {args.target_tokens}, 停止读取")
                            return

        collector = ContextCollector(model, tokenizer, sae, args.layer,
                                     sae_path=args.sae_path, device=args.device)
        n, path = collector.collect(
            read_texts(), output_path=args.output,
            threshold=args.threshold, batch_size=args.batch_size,
            selected_feature_ids=args.feature_ids,
        )
        print(f"✅ 收集完成: {n} 个特征 → {path}")

    elif args.cmd == "interpret":
        interp = FeatureInterpreter(
            api_key=args.api_key, model=args.llm_model, api_base=args.api_base,
        )
        summary = interp.run(args.context_path, args.output, sample_features=args.sample)
        print(f"✅ 解释完成: avg_score={summary['avg_score']}, scored={summary['features_scored']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
