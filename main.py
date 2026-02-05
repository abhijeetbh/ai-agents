from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()

class Response(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")
parser = PydanticOutputParser(pydantic_object=Response)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant that will help debug the provided source code. Answer the user request and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

query = input("How can I hlep you? ")
raw_response = agent_executor.invoke({"query":query})
print(raw_response)
print("-----------------------------------------------------------------------------------------")
structured_response = parser.parse(raw_response)
print(structured_response)