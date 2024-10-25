from dataclasses import dataclass, asdict
from typing import Optional
from json import dumps


@dataclass
class Media:
    isMedia: bool
    fileid: Optional[str]
    filename: Optional[str]
    mime_type: Optional[str]
    markdown: str = "NA"

    def __str__(self):
        if self.isMedia:
            return f"""
        <metadata>{self.filename}({self.mime_type})</metadata>
        <content>{self.markdown}</content>
        """
        else:
            return ""

    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__, ensure_ascii=False).encode('utf8')


@dataclass
class CompactMessage:
    identifier: str
    text: Optional[str]
    chattype: str
    chatid: int
    chatname: str
    userid: Optional[int]
    username: Optional[str]
    message_id: int
    created: Optional[str]
    lastUpdated: str
    edited: bool = False
    deleted: bool = False
    isForwarded: bool = False
    author: Optional[str] = None
    isBot: bool = False
    isAnswer: bool = False
    media: Optional[Media] = None

    def __str__(self):
        metadata = f"{self.username}@{self.chatname}@{self.lastUpdated}"
        if self.deleted:
            metadata += " (deleted)"
        elif self.edited:
            metadata += " (edited)"
        if self.isForwarded:
            metadata += f"\n\nForwarded from {self.author}"
        content = self.text
        if self.media.isMedia:
            content += f"\n\n{str(self.media)}"
        return f"{content}\n<metadata>{metadata}</metadata>"
    
    def to_dict(self):
        return self.__dict__

    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__, ensure_ascii=False).encode('utf8')

# END
