import os

from dotenv import load_dotenv
from openai import OpenAI
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()

## Web Search Agent
web_search_agent = Agent(name='web_search_agent', 
                         role="Search the web for information",
                         model=Groq(id="llama-3.2-1b-preview"),
                         tools=[DuckDuckGo], description="Search the web for information", 
                         instructions=["Always include sources"], 
                         show_tool_calls=True, 
                         markdown=True,
                         )

## Financial Agent
finance_agent = Agent(name='financial_agent', 
                        role="Financial analysis", 
                        model=Groq(id="llama-3.2-1b-preview"), 
                        tools=[YFinanceTools(stock_price=True, 
                                             analyst_recommendations=True, 
                                             stock_fundamentals=True,
                                             company_news=True),
                               ], 
                        description="Financial analysis", 
                        instructions=["Always include sources", "Use tables to display the data"], 
                        show_tool_calls=True, 
                        markdown=True,
                        )

## Add the agents to the list of agents


models = client.models.list()
for model in models.data:
    print(model.id)

multi_agent = Agent(
    model=Groq(id="llama-3.1-70b-versatile"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    markdown=True,
    show_tool_calls=True,
)

multi_agent.print_response("Summarise the analyst recommendation and share the latest news on AAPL?",
                           stream=True,
                           show_full_reasoning=True,
                           show_tool_calls=True,
                           ),
