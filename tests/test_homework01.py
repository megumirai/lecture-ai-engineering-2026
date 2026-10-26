"""第1回 宿題（app/ask.py と scripts/bench.py）のテスト。LLM も API キーも使わない。

偽のクライアントで、temperature の受け渡し、会話履歴の保持、計測値の取り出し、
レート制限時の再試行と記録を確かめる。
"""

from types import SimpleNamespace

import pytest
from google.genai import errors

from app import ask
from scripts import bench


def _response(text="ok", prompt=5, output=10, thoughts=None):
    usage = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=output,
        thoughts_token_count=thoughts,
        total_token_count=prompt + output + (thoughts or 0),
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def _api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": f"err{code}", "status": "X"}})


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # 返す応答か、投げる例外を順に
        self.calls = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    """google-genai の Chat と同じく、成功したときだけ履歴に積む。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.history = []

    def send_message(self, message):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.history += [("user", message), ("model", outcome.text)]
        return outcome

    def get_history(self):
        return list(self.history)


def _client(outcomes=(), chat_outcomes=()):
    chat = FakeChat(chat_outcomes)
    created = {}

    def create(*, model, config=None):
        created.update(model=model, config=config)
        return chat

    return SimpleNamespace(
        models=FakeModels(outcomes), chats=SimpleNamespace(create=create), chat=chat, created=created
    )


# --- app/ask.py ---------------------------------------------------------------


def test_generate_extracts_usage_and_passes_model():
    client = _client([_response("回答", prompt=7, output=685)])
    a = ask.generate(client, "LoRAとは？", model="m1")
    assert (a.text, a.model, a.prompt_tokens, a.output_tokens, a.total_tokens) == (
        "回答", "m1", 7, 685, 692,
    )
    assert a.elapsed_s >= 0
    assert client.models.calls[0].config is None  # 未指定ならモデルの既定値に任せる


def test_generate_passes_temperature_config():
    client = _client([_response()])
    ask.generate(client, "q", temperature=0.0)
    assert client.models.calls[0].config.temperature == 0.0


def test_missing_usage_metadata_does_not_crash():
    a = ask._to_answer(SimpleNamespace(text=None, usage_metadata=None), "m", 0.1)
    assert a.text == "" and a.prompt_tokens is None
    assert "入力: -" in a.usage_line()


@pytest.mark.parametrize("t", [-0.1, 2.1])
def test_temperature_out_of_range_is_rejected(t):
    with pytest.raises(ValueError):
        ask.build_config(t)


def test_chat_keeps_history_and_accumulates_tokens():
    client = _client(chat_outcomes=[_response("A1", prompt=5), _response("A2", prompt=40)])
    session = ask.ChatSession(client, model="m", temperature=1.0)
    assert client.created["config"].temperature == 1.0

    session.send("私の名前はMiraiです")
    session.send("私の名前は？")
    assert session.turns == 2
    assert client.chat.get_history()[0] == ("user", "私の名前はMiraiです")
    assert session.total_prompt_tokens == 45  # 履歴ごと送るので2回目の入力が増える


def test_run_chat_skips_blank_survives_api_error_and_exits():
    client = _client(chat_outcomes=[_api_error(429), _response("A")])
    session = ask.ChatSession(client)
    inputs = iter(["", "q1", "q2", "QUIT", "never"])
    out = []
    ask.run_chat(session, read=lambda _: next(inputs), write=out.append)

    joined = "\n".join(out)
    assert "レート制限" in joined  # エラーでも落ちずに続く
    assert "モデル> A" in joined
    assert session.turns == 1  # 失敗した q1 は履歴に残らない
    assert client.chat.get_history()[0] == ("user", "q2")


def test_run_chat_ends_on_eof():
    session = ask.ChatSession(_client())

    def eof(_):
        raise EOFError

    out = []
    ask.run_chat(session, read=eof, write=out.append)
    assert "0 往復" in out[-1]


# --- scripts/bench.py -------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_plan_covers_models_questions_and_temperatures():
    plan = bench.build_plan([bench.LITE, bench.FLASH], [0.0, 2.0], reps=3)
    model_trials = [t for t in plan if t.experiment == "model"]
    assert len(model_trials) == 2 * len(bench.QUESTIONS) * 3
    assert sum(t.model == bench.FLASH for t in plan) == 6  # 上位モデルの日次上限を食い潰さない
    assert {t.temperature for t in plan if t.experiment == "temperature"} == {0.0, 2.0}


def test_pacer_waits_per_model():
    clock = FakeClock()
    pacer = bench.Pacer({"a": 6, "b": 60}, clock=clock, sleep=clock.sleep)
    pacer.wait("a")
    pacer.wait("b")  # 別モデルは待たない
    assert clock.slept == []
    pacer.wait("a")
    assert clock.slept == [pytest.approx(10 * bench.SAFETY)]


def test_run_trial_retries_429_then_succeeds():
    client = _client([_api_error(429), _response("ok", output=3)])
    sleeps = []
    trial = bench.Trial("model", "m", "lora", None, 1)
    r = bench.run_trial(
        client, trial, bench.Pacer({"m": 10**9}), max_retries=2, backoff_s=5, sleep=sleeps.append
    )
    assert (r.status, r.attempts, r.output_tokens) == ("ok", 2, 3)
    assert sleeps == [5]


def test_run_trial_records_error_after_retries_exhausted():
    client = _client([_api_error(503)] * 3)
    trial = bench.Trial("model", "m", "lora", None, 1)
    r = bench.run_trial(
        client, trial, bench.Pacer({"m": 10**9}), max_retries=2, backoff_s=1, sleep=lambda s: None
    )
    assert (r.status, r.attempts, r.elapsed_s) == ("503", 3, None)


def test_run_trial_does_not_retry_client_errors():
    client = _client([_api_error(400)])
    trial = bench.Trial("model", "m", "lora", None, 1)
    r = bench.run_trial(
        client, trial, bench.Pacer({"m": 10**9}), max_retries=2, backoff_s=1, sleep=lambda s: None
    )
    assert (r.status, r.attempts) == ("400", 1)


def test_summary_aggregates_and_save_appends(tmp_path):
    base = {"experiment": "model", "model": "m", "question_id": "lora", "temperature": None}
    records = [
        bench.Record("t", rep=1, status="ok", attempts=1, elapsed_s=1.0, prompt_tokens=7,
                     output_tokens=100, text="回答1", **base),
        bench.Record("t", rep=2, status="ok", attempts=2, elapsed_s=3.0, prompt_tokens=7,
                     output_tokens=300, text="回答2", **base),
        bench.Record("t", rep=3, status="429", attempts=3, **base),
    ]
    md = bench.summarize(records)
    assert "| model | m | lora | 既定 | 2/3（429） | 3 | 2.00（1.00〜3.00） |" in md
    assert "200（100〜300）" in md and "rep3: （エラー 429）" in md

    bench.save(records[:1], tmp_path)
    bench.save(records[1:], tmp_path)  # 追記され、集計は全件から作り直される
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "2/3" in (tmp_path / "summary.md").read_text(encoding="utf-8")
