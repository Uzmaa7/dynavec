"""Tests for Human-in-the-loop / interrupts functionality in dynavec agents."""

import asyncio
from collections.abc import AsyncIterator, Iterator

from dynavec.agents.base import interrupt, tool
from dynavec.agents.react import ReActAgent
from dynavec.chat.base import ChatModel, ChatResult, Message, ToolCall


class MockChatModel(ChatModel):
    """Mock model that returns predefined tool calls or responses sequentially."""

    def __init__(self, responses: list[ChatResult]) -> None:
        super().__init__()
        self.model_name = "mock-model"
        self.responses = responses
        self.call_count = 0

    def invoke(self, messages: list[Message], tools=None, **kwargs) -> ChatResult:
        res = self.responses[self.call_count]
        self.call_count += 1
        return res

    async def ainvoke(self, messages: list[Message], tools=None, **kwargs) -> ChatResult:
        return self.invoke(messages, tools=tools, **kwargs)

    def stream(self, messages: list[Message], tools=None, **kwargs) -> Iterator[ChatResult]:
        res = self.invoke(messages, tools=tools, **kwargs)
        yield res

    async def astream(
        self, messages: list[Message], tools=None, **kwargs
    ) -> AsyncIterator[ChatResult]:
        res = await self.ainvoke(messages, tools=tools, **kwargs)
        yield res


@tool
def sensitive_action_tool(action: str) -> str:
    """A tool that requires human approval before proceeding."""
    interrupt(
        thread_id="thread-123",
        node_id="approval_node",
        payload={"action": action, "requires_approval": True},
    )
    return f"Executed {action}"


def test_interrupt_raised_and_caught():
    """Test that NodeInterrupt pauses execution and populates interrupt_payload."""
    mock_responses = [
        ChatResult(
            message=Message(
                role="assistant",
                content="Requesting sensitive action.",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="sensitive_action_tool",
                        arguments={"action": "delete_database"},
                    )
                ],
            )
        )
    ]
    model = MockChatModel(mock_responses)
    agent = ReActAgent(model=model, tools=[sensitive_action_tool])

    result = agent.run("Delete the production database", thread_id="thread-123")

    assert result.finished is False
    assert result.termination_reason == "interrupted"
    assert result.interrupt_payload is not None
    assert result.interrupt_payload["thread_id"] == "thread-123"
    assert result.interrupt_payload["node_id"] == "approval_node"
    assert result.interrupt_payload["payload"]["action"] == "delete_database"


def test_resume_execution():
    """Test resuming an interrupted agent with human decision."""
    initial_responses = [
        ChatResult(
            message=Message(
                role="assistant",
                content="Requesting sensitive action.",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="sensitive_action_tool",
                        arguments={"action": "delete_database"},
                    )
                ],
            )
        )
    ]
    model = MockChatModel(initial_responses)
    agent = ReActAgent(model=model, tools=[sensitive_action_tool])

    result = agent.run("Delete database", thread_id="thread-123")
    assert result.termination_reason == "interrupted"

    # Simulate resume after human approves
    resume_responses = [
        ChatResult(
            message=Message(
                role="assistant",
                content="Action was approved by human. Proceeding completed successfully.",
            )
        )
    ]
    resume_model = MockChatModel(resume_responses)
    resume_agent = ReActAgent(model=resume_model, tools=[sensitive_action_tool])

    saved_messages = result.interrupt_payload["messages"]
    start_step = result.interrupt_payload["start_step"]

    final_result = resume_agent.resume(
        messages=saved_messages,
        human_decision={"approved": True},
        thread_id="thread-123",
        start_step=start_step,
    )

    assert final_result.finished is True
    assert final_result.termination_reason == "completed"


def test_async_interrupt_and_resume():
    """Test async arun and aresume functionality using asyncio.run."""

    async def _run_async():
        initial_responses = [
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Requesting sensitive action.",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="sensitive_action_tool",
                            arguments={"action": "deploy_code"},
                        )
                    ],
                )
            )
        ]
        model = MockChatModel(initial_responses)
        agent = ReActAgent(model=model, tools=[sensitive_action_tool])

        result = await agent.arun("Deploy code to production", thread_id="thread-456")
        assert result.termination_reason == "interrupted"

        resume_responses = [
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Deployment finished after human confirmation.",
                )
            )
        ]
        resume_agent = ReActAgent(
            model=MockChatModel(resume_responses), tools=[sensitive_action_tool]
        )

        final_result = await resume_agent.aresume(
            messages=result.interrupt_payload["messages"],
            human_decision="Approved by DevOps lead",
            thread_id="thread-456",
            start_step=result.interrupt_payload["start_step"],
        )

        assert final_result.finished is True
        assert final_result.termination_reason == "completed"

    asyncio.run(_run_async())
