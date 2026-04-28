from cosmere_rag.agent.types import AgentResponse, Citation
from cosmere_rag.slack.formatting import agent_response_to_blocks


def _resp(**kwargs) -> AgentResponse:
    base = dict(answer="Kelsier was a Mistborn.", citations=[], trace_url=None)
    base.update(kwargs)
    return AgentResponse(**base)


def test_section_block_present():
    blocks = agent_response_to_blocks(_resp())
    assert blocks[0]["type"] == "section"
    assert blocks[0]["text"]["text"] == "Kelsier was a Mistborn."


def test_citations_render_as_context_block():
    response = _resp(
        citations=[
            Citation(title="Kelsier", url="https://coppermind.net/wiki/Kelsier"),
            Citation(title="Vin", url="https://coppermind.net/wiki/Vin"),
        ]
    )
    blocks = agent_response_to_blocks(response)
    assert len(blocks) == 2
    assert blocks[1]["type"] == "context"
    text = blocks[1]["elements"][0]["text"]
    assert "<https://coppermind.net/wiki/Kelsier|Kelsier>" in text
    assert "<https://coppermind.net/wiki/Vin|Vin>" in text


def test_no_citation_block_when_empty():
    blocks = agent_response_to_blocks(_resp())
    assert len(blocks) == 1


def test_trace_url_gated_off_by_default():
    response = _resp(trace_url="https://smith.langchain.com/run/abc")
    blocks = agent_response_to_blocks(response)
    assert all("trace" not in str(b) for b in blocks)


def test_trace_url_included_when_enabled():
    response = _resp(trace_url="https://smith.langchain.com/run/abc")
    blocks = agent_response_to_blocks(response, include_trace_url=True)
    last = blocks[-1]
    assert last["type"] == "context"
    assert "smith.langchain.com/run/abc" in last["elements"][0]["text"]


def test_empty_answer_falls_back_to_placeholder():
    blocks = agent_response_to_blocks(_resp(answer=""))
    assert blocks[0]["text"]["text"] == "_no answer_"
