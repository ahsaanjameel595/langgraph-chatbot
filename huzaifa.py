from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


# ------------------ Model ------------------

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)


# ------------------ Schema ------------------

class Facts(BaseModel):
    fact_1: str = Field(
        description="Fact 1 about the topic"
    )

    fact_2: str = Field(
        description="Fact 2 about the topic"
    )

    fact_3: str = Field(
        description="Fact 3 about the topic"
    )


# ------------------ Parser ------------------

parser = PydanticOutputParser(
    pydantic_object=Facts
)


# ------------------ Prompt ------------------

template = PromptTemplate(
    template="""
Give me three facts about {topic}.

{format_instruction}
""",

    input_variables=["topic"],

    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)


# ------------------ Create Prompt ------------------

prompt = template.invoke({
    "topic": "Black Hole"
})

print(prompt.text)


# ------------------ Model Response ------------------

result = model.invoke(prompt)

print("\nRaw Response:")
print(result.content)


# ------------------ Parse Response ------------------

final_result = parser.parse(result.content)

print("\nStructured Output:")
print(final_result)