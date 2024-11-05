import os
import json
from enum import Enum
from dataclasses import dataclass
import openai
import numpy as np

from .agents import BaseTool, Encoder

ROUTER_AGENT_PROMPT = """
**Prompt:**

Given a user query, determine whether the query can be answered by the given agent. 

**Input:**

* **query:** The user's query as a string.
* **agent_description:** The agent's description as a string.

**Output:**

* **score:** A float between -1 and 1, indicating the relevance of the agent.

**Example:**

**Query:** "Find me the latest document about 'AI safety' in the 'research' namespace, top 3 results."
**Agent Description:** 
<AGENT_DESCRIPTION>
Description: Execute web searches via DuckDuckGo, collect factual information from reliable online sources

Parameters:

`query` (string): The query to be executed on DuckDuckGo. Constraint: 1 <= len(query) <= 200
`top_n` (int): The number of results to return. Default is 5. Constraint: 1 <= top_n <= 10

Returns:

`response` (dict): {
    "result": [
                (
                    "web_search_result", 
                    {"title": "{TITLE}", "href": "{URL}", "body": "{WEBPAGE_SUMMARY}", "text": "{WEBPAGE_CONTENT}"}, 
                    "{TIMESTAMP}", 
                    True
                )
            ]
        }
`next_func` (string): The name of the next function to call.

Keywords: web, internet, online, search, Wikipedia, article, webpage, website, URL, DuckDuckGo, browser, HTTP, link, database, reference, citation, source, fact-check, current, updated, recent, global, worldwide, information

Relevant Query Types:

"What is the latest information about..."
"Find articles about..."
"Search for facts regarding..."
"What does Wikipedia say about..."
"Look up current details on..."

</AGENT_DESCRIPTION>

**Output:**

{
  "score": 0.8
}

"""

QUERY_TEMPLATE = """
**Query:** <QUERY></QUERY>
**Agent Description:** <AGENT_DESCRIPTION></AGENT_DESCRIPTION>
"""


class RoutingMethod(Enum):
    SEMANTIC = "semantic"
    LLM = "llm"
    HYBRID = "hybrid"


class Weights:
    def __init__(self, semantic: float = 0.5, llm: float = 0.5):
        self.semantic = semantic
        self.llm = llm

    def apply_weights(self, semantic_score: float, llm_score: float) -> float:
        return self.semantic * semantic_score + self.llm * llm_score


@dataclass
class RoutingScoreValue:
    semantic_score: float = 0.0
    llm_score: float = 0.0


@dataclass
class RoutingScore:
    # Positive value is desired
    key: str
    value: RoutingScoreValue
    weights: Weights = Weights(0.5, 0.5)

    @property
    def total_score(self) -> float:
        return self.weights.apply_weights(self.value.semantic_score, self.value.llm_score)


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Naive approach to compute the cosine similarity between two vectors of potentially different length.

    Args:
    - v1: First vector
    - v2: Second vector

    Returns:
    - Cosine similarity between the two vectors (-1.0, 1.0)

    Notes:
    - Zero-padding for different lengths
    - Future enhancement: Sliding window with pooling
    """
    max_length = max(len(v1), len(v2))
    if len(v1) < max_length:
        v1 = np.pad(v1, (0, max_length - len(v1)), 'constant', constant_values=0.0)
    if len(v2) < max_length:
        v2 = np.pad(v2, (0, max_length - len(v2)), 'constant', constant_values=0.0)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class AgentsRouter:
    def __init__(
            self, instruction: str,
            agents: list[BaseTool], encoder: Encoder = Encoder(),
            method: RoutingMethod = RoutingMethod.SEMANTIC,
            weights: Weights = Weights(0.5, 0.5),
            threshold: float = 1.0, gpt_model_name: str = "gpt-4o-mini-2024-07-18"
    ):
        self.agents = dict()
        self.routing_method = method
        self.weights = weights
        if self.routing_method == RoutingMethod.SEMANTIC:
            self.weights.llm = 0.0
        elif self.routing_method == RoutingMethod.LLM:
            self.weights.semantic = 0.0
        self.threshold = threshold
        self.encoder = encoder
        self.init(agents)
        self.gpt_model_name = gpt_model_name
        if instruction is None or instruction == "":
            self.instruction = ROUTER_AGENT_PROMPT

    def init(self, agents: list[BaseTool]):
        for agent in agents:
            embedding = self.encoder.text_to_embedding(agent.description)
            agent_value = {"agent": agent, "embedding": np.array(embedding)}
            self.agents[agent.name] = agent_value

    def route(self, params: str) -> list[BaseTool]:
        by_semantic_matching = self.semantic_matching(params)
        # print(f"Semantic: {str(by_semantic_matching)}")
        by_llm_matching = self.llm_matching(params)
        # print(f"LLM: {str(by_llm_matching)}")
        agents = dict()
        for k, v in by_semantic_matching.items():
            agents[k] = RoutingScore(key=k, value=RoutingScoreValue(semantic_score=v), weights=self.weights)
        for k, v in by_llm_matching.items():
            if k not in agents:
                agents[k] = RoutingScore(key=k, value=RoutingScoreValue(llm_score=v), weights=self.weights)
            else:
                agents[k].value.llm_score += v
        recommended_agents = agents.keys()
        output = []
        for agent_name, agent in self.agents.items():
            if agent_name in recommended_agents:
                score = agents[agent_name].total_score
                print(f"Agent {agent_name} score: {score}")
                if score >= self.threshold:
                    output.append(agent["agent"])
        return output

    def semantic_matching(self, query: str) -> dict[str, float]:
        query_embedding = np.array(self.encoder.text_to_embedding(query))
        # print("Generated query embedding!")
        for k, v in self.agents.items():
            semantic_score = compute_cosine_similarity(v["embedding"], query_embedding)
            v["semantic_score"] = semantic_score
        return {k: v["semantic_score"] for k, v in self.agents.items()}

    def llm_matching(self, query: str) -> dict[str, float]:
        output = dict()
        for k, v in self.agents.items():
            agent: BaseTool = v["agent"]
            formatted_query = QUERY_TEMPLATE.replace(
                "<QUERY></QUERY>", f"<QUERY>{query}</QUERY>"
            )
            formatted_query = formatted_query.replace(
                "<AGENT_DESCRIPTION></AGENT_DESCRIPTION>",
                f"<AGENT_DESCRIPTION>{agent.description}</AGENT_DESCRIPTION>"
            )
            messages = [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": formatted_query}
            ]
            try:
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                response = client.chat.completions.create(
                    model=self.gpt_model_name,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.7,
                    n=1
                )
                result = response.choices[0].message.content.strip()
                json_body = json.loads(result)
                llm_score = float(json_body["score"])
                if llm_score < -1.0 or llm_score > 1.0:
                    raise Exception(f"LLM score is not in range: {llm_score}")
                output[k] = llm_score
            except json.JSONDecodeError as json_loads_error:
                print(f"Error: {json_loads_error}")
            except Exception as e:
                print(f"Error: {e}")
        return output
