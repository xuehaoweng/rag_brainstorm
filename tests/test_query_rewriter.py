from app.retrieval.query_rewriter import rewrite_query, _parse_rewrites


class FakeRewriteLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "项目安装与配置步骤\n"
            "生产环境部署指南\n"
            "服务启动和运行方式"
        )


class NumberedRewriteLLM:
    """Simulates LLM that outputs numbered lines despite instructions."""
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "1. 项目安装与配置步骤\n"
            "2) 生产环境部署指南\n"
            "- 服务启动和运行方式"
        )


class FailingLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM unavailable")


class EmptyLLM:
    name = "fake"
    model = "fake-model"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "\n\n  \n"


def test_rewrite_query_returns_original_plus_rewrites() -> None:
    result = rewrite_query("怎么部署？", FakeRewriteLLM())

    assert result[0] == "怎么部署？"  # original always first
    assert len(result) == 4  # original + 3 rewrites
    assert "项目安装与配置步骤" in result
    assert "生产环境部署指南" in result
    assert "服务启动和运行方式" in result


def test_rewrite_query_strips_numbering() -> None:
    result = rewrite_query("怎么部署？", NumberedRewriteLLM())

    assert result[0] == "怎么部署？"
    assert "项目安装与配置步骤" in result
    assert "生产环境部署指南" in result
    assert "服务启动和运行方式" in result
    # No numbering prefixes in results
    for q in result[1:]:
        assert not q.startswith("1")
        assert not q.startswith("2")
        assert not q.startswith("-")


def test_rewrite_query_fallback_on_llm_failure() -> None:
    result = rewrite_query("怎么部署？", FailingLLM())

    assert result == ["怎么部署？"]


def test_rewrite_query_fallback_on_empty_response() -> None:
    result = rewrite_query("怎么部署？", EmptyLLM())

    assert result == ["怎么部署？"]


def test_rewrite_query_respects_num_rewrites() -> None:
    result = rewrite_query("怎么部署？", FakeRewriteLLM(), num_rewrites=2)

    assert result[0] == "怎么部署？"
    assert len(result) == 3  # original + 2


def test_parse_rewrites_handles_mixed_formats() -> None:
    raw = """1. 查询一
2) 查询二
- 查询三
* 查询四

查询五"""
    result = _parse_rewrites(raw)
    assert result == ["查询一", "查询二", "查询三", "查询四", "查询五"]


def test_parse_rewrites_empty_input() -> None:
    assert _parse_rewrites("") == []
    assert _parse_rewrites("  \n  \n  ") == []
