import logging
from telegram import Message
from typing import Optional, List

logger = logging.getLogger(__name__)


def to_display(data: dict) -> str:
    assert not data["deleted"]
    return (
        f"\n<strong>{data['username']}</strong> => [{data['text']}]@{data['lastUpdated']}"
        + (" (edited)" if data["edited"] else "")
    )
