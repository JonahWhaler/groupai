import os

master: int = int(os.environ["MASTER_TLG_ID"])
assert master != 0
