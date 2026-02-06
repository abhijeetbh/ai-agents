from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Response(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
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

query = input("How can I help you? ")
raw_response = agent_executor.invoke({"query":query})
print(raw_response)
print("-----------------------------------------------------------------------------------------")
structured_response = parser.parse(str(raw_response))
print(structured_response)