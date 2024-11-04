from typing import Protocol


class CallableTool(Protocol):
    @property
    def priority(self) -> int:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        ...
    
    @property
    def next_tool_name(self) -> str | None:
        ...
    
    async def __call__(self, params: str) -> dict:
        ...

    def validate(self, params: str) -> bool:
        ...
