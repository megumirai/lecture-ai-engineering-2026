"""第1回 宿題（基礎・発展）: 同じ質問を条件を変えて繰り返し、回答・応答時間・トークン数を記録する。

使い方:
    uv run python -m scripts.bench                 # 既定の計画（下の PLAN）を実行
    uv run python -m scripts.bench --dry-run       # API を呼ばずに、実行する組み合わせと所要時間の目安だけ表示
    uv run python -m scripts.bench --burst 20      # 待たずに連続で呼び、どこで 429 になるかを観察（lite のみ）

出力（--out、既定 lectures/01/results/）:
    runs.jsonl   1回の呼び出しごとの生データ（回答全文を含む）
    summary.md   条件ごとの集計表と回答の抜粋（answer.md に貼る元）

無料枠の上限（講座の実測、2026-10）に合わせて、モデルごとに呼び出し間隔を空ける。
    gemini-3.5-flash-lite : 毎分 15 / 1日 500
    gemini-3.8-flash      : 毎分 5  / 1日 10〜20 前後（503 が返ることもある）
既定の計画は 3.8-flash を 6 回だけ使う。上限は変わりうるので AI Studio で確認すること。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION, build_config, generate
from app.config import get_api_key

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.8-flash"

QUESTIONS = {
    "lora": "LoRAとは何ですか？",
    "levels": DEFAULT_QUESTION,
}

# 1分あたりの上限（RPM）。表にないモデルは慎重側の 5 とする
RPM = {LITE: 15, FLASH: 5}
DEFAULT_RPM = 5
SAFETY = 1.15  # 上限ちょうどを狙わず、15% 余裕を持たせる

RETRYABLE = {429, 503}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "lectures" / "01" / "results"


@dataclass(frozen=True)
class Trial:
    experiment: str  # "model" または "temperature"
    model: str
    question_id: str
    temperature: float | None
    rep: int


@dataclass
class Record:
    timestamp: str
    experiment: str
    model: str
    question_id: str
    temperature: float | None
    rep: int
    status: str  # "ok" または HTTP ステータス
    attempts: int
    elapsed_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None
    text: str = ""
    error: str = ""


def build_plan(models: list[str], temperatures: list[float], reps: int) -> list[Trial]:
    """基礎（モデル比較・temperature 比較）と発展（時間・トークン比較）を兼ねた試行の一覧。"""
    plan = [
        Trial("model", m, q, None, r)
        for r in range(1, reps + 1)
        for m in models
        for q in QUESTIONS
    ]
    plan += [
        Trial("temperature", LITE, "levels", t, r)
        for r in range(1, reps + 1)
        for t in temperatures
    ]
    return plan


class Pacer:
    """モデルごとに、前回の呼び出しから一定時間空ける。clock と sleep はテスト用に差し替えられる。"""

    def __init__(
        self,
        rpm: dict[str, int],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def interval(self, model: str) -> float:
        return 60.0 / self._rpm.get(model, DEFAULT_RPM) * SAFETY

    def wait(self, model: str) -> None:
        if model in self._last:
            remaining = self._last[model] + self.interval(model) - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last[model] = self._clock()


def run_trial(
    client: genai.Client,
    trial: Trial,
    pacer: Pacer,
    *,
    max_retries: int,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Record:
    """1試行を実行する。429/503 は指数バックオフで再試行し、最終結果と試行回数を記録する。"""
    question = QUESTIONS[trial.question_id]
    base = {
        "experiment": trial.experiment,
        "model": trial.model,
        "question_id": trial.question_id,
        "temperature": trial.temperature,
        "rep": trial.rep,
    }
    attempt = 0
    while True:
        attempt += 1
        pacer.wait(trial.model)
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            a = generate(client, question, model=trial.model, temperature=trial.temperature)
        except errors.APIError as e:
            if e.code in RETRYABLE and attempt <= max_retries:
                sleep(backoff_s * 2 ** (attempt - 1))
                continue
            return Record(stamp, status=str(e.code), attempts=attempt, error=e.message or "", **base)
        return Record(
            stamp,
            status="ok",
            attempts=attempt,
            elapsed_s=round(a.elapsed_s, 3),
            prompt_tokens=a.prompt_tokens,
            output_tokens=a.output_tokens,
            thoughts_tokens=a.thoughts_tokens,
            total_tokens=a.total_tokens,
            text=a.text,
            **base,
        )


def run_burst(client: genai.Client, n: int, model: str = LITE) -> list[Record]:
    """間隔を空けずに n 回呼び、何回目で 429 になるかを見る。再試行はしない。"""
    records = []
    for i in range(1, n + 1):
        trial = Trial("burst", model, "lora", None, i)
        records.append(run_trial(client, trial, Pacer({model: 10**9}), max_retries=0, backoff_s=0))
        print(f"  burst {i:>3}: {records[-1].status}")
    return records


def _fmt(values: Iterable[float | int | None], digits: int = 0) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return "-"
    mean = statistics.fmean(xs)
    return f"{mean:.{digits}f}（{min(xs):.{digits}f}〜{max(xs):.{digits}f}）"


def summarize(records: list[Record]) -> str:
    """条件ごとの集計表（平均と最小〜最大）と、回答の抜粋を Markdown で返す。"""
    groups: dict[tuple, list[Record]] = {}
    for r in records:
        groups.setdefault((r.experiment, r.model, r.question_id, r.temperature), []).append(r)

    lines = [
        "| 実験 | モデル | 質問 | temperature | 成功/試行 | 再試行 | 応答時間 s | 入力 | 出力 | 思考 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (exp, model, qid, temp), rs in groups.items():
        ok = [r for r in rs if r.status == "ok"]
        retries = sum(r.attempts - 1 for r in rs)
        errs = sorted({r.status for r in rs if r.status != "ok"})
        status = f"{len(ok)}/{len(rs)}" + (f"（{', '.join(errs)}）" if errs else "")
        lines.append(
            f"| {exp} | {model} | {qid} | {'既定' if temp is None else temp} | {status} | {retries}"
            f" | {_fmt((r.elapsed_s for r in ok), 2)} | {_fmt(r.prompt_tokens for r in ok)}"
            f" | {_fmt(r.output_tokens for r in ok)} | {_fmt(r.thoughts_tokens or 0 for r in ok)} |"
        )

    lines += ["", "## 回答の抜粋（先頭 120 字）", ""]
    for (exp, model, qid, temp), rs in groups.items():
        if exp == "burst":
            continue
        lines.append(f"### {exp} / {model} / {qid} / temperature={'既定' if temp is None else temp}")
        for r in rs:
            body = " ".join(r.text.split())[:120] if r.status == "ok" else f"（エラー {r.status}）"
            lines.append(f"- rep{r.rep}: {body}")
        lines.append("")
    return "\n".join(lines)


def save(records: list[Record], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    all_records = [
        Record(**json.loads(line))
        for line in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (out / "summary.md").write_text(
        "# 第1回 宿題 計測結果\n\n" + summarize(all_records) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="条件を変えて同じ質問を繰り返し、記録する")
    parser.add_argument("--models", nargs="+", default=[LITE, FLASH])
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--backoff", type=float, default=20.0, help="再試行の初回待ち秒（倍々に延ばす）")
    parser.add_argument("--burst", type=int, default=0, help="レート制限の観察（lite に連続 n 回）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.reps < 1:
        parser.error("--reps は 1 以上")
    for t in args.temperatures:
        build_config(t)  # 範囲外なら ValueError で止める

    plan = [] if args.burst else build_plan(args.models, args.temperatures, args.reps)
    pacer = Pacer(RPM)
    calls = {m: sum(t.model == m for t in plan) for m in {t.model for t in plan}}
    eta = max((n - 1) * pacer.interval(m) for m, n in calls.items()) if calls else 0
    print(f"試行 {len(plan) or args.burst} 回  モデル別 {calls or {LITE: args.burst}}  待ち時間の目安 {eta:.0f}s")
    if args.dry_run:
        for t in plan:
            print(f"  {t}")
        return

    client = genai.Client(api_key=get_api_key())
    if args.burst:
        records = run_burst(client, args.burst)
    else:
        records = []
        for i, trial in enumerate(plan, 1):
            r = run_trial(client, trial, pacer, max_retries=args.max_retries, backoff_s=args.backoff)
            records.append(r)
            took = "-" if r.elapsed_s is None else f"{r.elapsed_s:.2f}s"
            print(f"[{i:>2}/{len(plan)}] {r.model} {r.question_id} t={r.temperature} rep{r.rep}: "
                  f"{r.status} {took} out={r.output_tokens}")
    save(records, args.out)
    print(f"\n保存: {args.out / 'runs.jsonl'} / {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
