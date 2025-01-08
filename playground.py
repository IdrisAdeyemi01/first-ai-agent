import os

import openai
import phi
import phi.api as api
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

load_dotenv()

# client = openai.ChatCompletion(api_key=os.getenv("OPENAI_API_KEY"))

api=os.getenv("PHI_API_KEY")

## Web Search Agent
web_search_agent = Agent(name='web_search_agent', 
                         role="Search the web for information",
                         model=Groq(id="llama3-70b-8192"),
                         tools=[DuckDuckGo()], description="Search the web for information", 
                         instructions=["Always include sources"], 
                         show_tool_calls=True, 
                         markdown=True,
                         )

## Financial Agent
finance_agent = Agent(name='financial_agent', 
                        role="Financial analysis", 
                        model=Groq(id="llama3-70b-8192"), 
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



app = Playground(
    agents=[web_search_agent, finance_agent],
).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)

