"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask                          # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --temperature 0 "..."    # temperature を指定する（宿題・基礎）
    uv run python -m app.ask --chat                   # 会話履歴を保ったまま続けて質問する（宿題・実装）

単発の呼び出しはツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

各呼び出しの応答時間とトークン数を計測して表示する（宿題・発展。集計は scripts/bench.py）。
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# 会話モードを終了する入力（大文字小文字は区別しない）
EXIT_COMMANDS = frozenset({"exit", "quit", "q", ":q"})

# Gemini API が受け付ける temperature の範囲
TEMPERATURE_RANGE = (0.0, 2.0)


@dataclass(frozen=True)
class Answer:
    """1回の呼び出し結果。トークン数は API が返さなかった場合 None。"""

    text: str
    model: str
    elapsed_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    thoughts_tokens: int | None
    total_tokens: int | None

    def usage_line(self) -> str:
        def fmt(n: int | None) -> str:
            return "-" if n is None else str(n)

        return (
            f"[tokens] 入力: {fmt(self.prompt_tokens)}"
            f" / 出力: {fmt(self.output_tokens)}"
            f" / 思考: {self.thoughts_tokens or 0}"
            f" / 合計: {fmt(self.total_tokens)}"
            f"  [time] {self.elapsed_s:.2f}s"
        )


def build_config(temperature: float | None) -> types.GenerateContentConfig | None:
    """temperature が指定されたときだけ config を作る（未指定ならモデルの既定値に任せる）。"""
    if temperature is None:
        return None
    low, high = TEMPERATURE_RANGE
    if not low <= temperature <= high:
        raise ValueError(f"temperature は {low}〜{high} の範囲で指定してください: {temperature}")
    return types.GenerateContentConfig(temperature=temperature)


def _to_answer(response: Any, model: str, elapsed_s: float) -> Answer:
    usage = getattr(response, "usage_metadata", None)

    def count(name: str) -> int | None:
        return getattr(usage, name, None) if usage is not None else None

    return Answer(
        text=response.text or "",
        model=model,
        elapsed_s=elapsed_s,
        prompt_tokens=count("prompt_token_count"),
        output_tokens=count("candidates_token_count"),
        thoughts_tokens=count("thoughts_token_count"),
        total_tokens=count("total_token_count"),
    )


def generate(
    client: genai.Client,
    question: str,
    *,
    model: str = MODEL,
    temperature: float | None = None,
) -> Answer:
    """1回だけ質問し、応答時間とトークン数を添えて返す。APIエラーは呼び出し側に任せる。"""
    config = build_config(temperature)
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question, config=config)
    return _to_answer(response, model, time.perf_counter() - start)


def ask(question: str, *, model: str = MODEL, temperature: float | None = None) -> str:
    """単発で質問して回答テキストを返す（第1回 演習の入口。トークン数も表示する）。"""
    answer = generate(
        genai.Client(api_key=get_api_key()), question, model=model, temperature=temperature
    )
    print(answer.usage_line())
    return answer.text


class ChatSession:
    """会話履歴を保ったまま続けて質問する（宿題・実装）。

    google-genai の Chat は、送信が成功したときだけ質問と回答を履歴に積む。
    毎回、それまでの履歴全体がモデルに送られるため、入力トークンはターンごとに増える。
    """

    def __init__(
        self, client: genai.Client, *, model: str = MODEL, temperature: float | None = None
    ) -> None:
        self.model = model
        self._chat = client.chats.create(model=model, config=build_config(temperature))
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0

    def send(self, message: str) -> Answer:
        start = time.perf_counter()
        response = self._chat.send_message(message)
        answer = _to_answer(response, self.model, time.perf_counter() - start)
        self.total_prompt_tokens += answer.prompt_tokens or 0
        self.total_output_tokens += answer.output_tokens or 0
        return answer

    @property
    def turns(self) -> int:
        """履歴に積まれた往復の数（ユーザーとモデルで1往復）。"""
        return len(self._chat.get_history()) // 2


def run_chat(
    session: ChatSession,
    *,
    first_message: str | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    """対話ループ。exit / quit / 空行以外の入力を送り、Ctrl+D・Ctrl+C でも終了する。"""
    write(f"[model] {session.model}  （終了: {' / '.join(sorted(EXIT_COMMANDS))}、Ctrl+D）")
    pending = first_message
    while True:
        if pending is None:
            try:
                pending = read("\nあなた> ")
            except (EOFError, KeyboardInterrupt):
                write("")
                break
        message, pending = pending.strip(), None
        if not message:
            continue
        if message.lower() in EXIT_COMMANDS:
            break
        try:
            answer = session.send(message)
        except errors.APIError as e:
            # 失敗した質問は履歴に積まれないので、そのまま続けられる
            write(api_error_message(e.code, e.message or ""))
            continue
        write(f"\nモデル> {answer.text}")
        write(f"{answer.usage_line()}  [履歴] {session.turns} 往復")
    write(
        f"[会話の合計] {session.turns} 往復 / 入力 {session.total_prompt_tokens}"
        f" / 出力 {session.total_output_tokens} トークン"
    )


def _temperature(value: str) -> float:
    try:
        t = float(value)
        build_config(t)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    parser.add_argument("--temperature", type=_temperature, default=None, help="0.0〜2.0")
    parser.add_argument("--model", default=MODEL, help=f"既定: {MODEL}（環境変数 AIE_MODEL）")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())

    if args.chat:
        session = ChatSession(client, model=args.model, temperature=args.temperature)
        run_chat(session, first_message=args.question)
        return

    question = args.question or DEFAULT_QUESTION
    temp_label = "既定" if args.temperature is None else args.temperature
    print(f"[model] {args.model}  [temperature] {temp_label}")
    print(f"[質問] {question}\n")
    try:
        answer = generate(client, question, model=args.model, temperature=args.temperature)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message or ""))
        raise SystemExit(1)
    print(answer.text)
    print(answer.usage_line())


if __name__ == "__main__":
    main()
