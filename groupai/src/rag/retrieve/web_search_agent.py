import json
import aiohttp
import asyncio
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import random

WEB_EXPLORER_AGENT_PROMPT = """
Description: Execute web searches via DuckDuckGo, collect factual information from reliable online sources

Parameters:

`query` (string): The query to be executed on DuckDuckGo. Constraint: 1 <= len(query) <= 200
`top_n` (int): The number of results to return. Default is 5. Constraint: 1 <= top_n <= 10

Returns:

`response` (dict): {"result": [{"title": "{TITLE}", "href": "{URL}", "body": "{WEBPAGE_SUMMARY}", "text": "{WEBPAGE_CONTENT}"}]}
`next_func` (string): The name of the next function to call.

Keywords: web, internet, online, search, Wikipedia, article, webpage, website, URL, DuckDuckGo, browser, HTTP, link, database, reference, citation, source, fact-check, current, updated, recent, global, worldwide, information

Relevant Query Types:

"What is the latest information about..."
"Find articles about..."
"Search for facts regarding..."
"What does Wikipedia say about..."
"Look up current details on..."
"""

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0'
]


class DuckDuckGoSearchAgent:
    def __init__(self, priority=1, next_func: str | None = None, safesearch="moderate", region="my-en", pause=0.5):
        self.__name = "DuckDuckGoSearchAgent"
        self.__safesearch = safesearch  # on, moderate, off
        self.__region = region
        self.__pause = pause
        self.__priority = priority
        self.__next_func = next_func

    @property
    def priority(self) -> int:
        return self.__priority

    @property
    def prompt(self) -> str:
        return WEB_EXPLORER_AGENT_PROMPT

    @property
    def random_user_agent(self) -> str:
        return random.choice(user_agents)

    @property
    def headers(self) -> dict:
        return {
            'User-Agent': self.random_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

    @classmethod
    def validate(cls, params: str) -> bool:
        params = json.loads(params)
        query = params.get("query", None)
        top_n = params.get("top_n", 5)
        if query is None:
            return False
        query = query.strip()
        conditions = [len(query) > 0, len(query) <=
                      200, top_n >= 1, top_n <= 10]
        if not all(conditions):
            return False
        return True

    async def fetch_data(self, session, url):
        try:
            await asyncio.sleep(self.__pause)
            async with session.get(url, headers=self.headers) as response:
                data = await response.text()
                soup = BeautifulSoup(data, 'html.parser')
                return soup.find('body').text
        except Exception as e:
            return ""

    async def __call__(self, params: str) -> tuple[dict, str | None]:
        params = json.loads(params)
        query = params.get("query", None)
        top_n = params.get("top_n", 5)
        print(f"Begin Execution: {self.__name}...")
        output = dict()
        with DDGS() as ddgs:
            top_search = []
            for r in ddgs.text(keywords=query, region=self.__region, safesearch=self.__safesearch, max_results=top_n):
                top_search.append(r)
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_data(session, r['href']) for r in top_search]
            search_results = await asyncio.gather(*tasks)
            for r, sr in zip(top_search, search_results):
                r['html'] = sr
            output["result"] = top_search
        print(f"End Execution: {self.__name}")
        return output, self.__next_func
