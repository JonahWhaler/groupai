import logging
import chromadb

logger = logging.getLogger(__name__)


def to_display(data: dict) -> str:
    assert not data["deleted"]
    return (
        f"\n<strong>{data['username']}</strong> => [{data['text']}]@{data['lastUpdated']}"
        + (" (edited)" if data["edited"] else "")
    )


class ChromaDBFactory:
    instance: chromadb.ClientAPI | None = None

    @classmethod
    def get_instance(
        cls, persist: bool | None, persist_directory: str | None
    ) -> chromadb.ClientAPI:
        if cls.instance:
            return cls.instance
        if persist and persist_directory:
            cls.instance = chromadb.Client(
                settings=chromadb.Settings(
                    is_persistent=persist,
                    persist_directory=persist_directory,
                )
            )
        else:
            cls.instance = chromadb.Client()
        return cls.instance
