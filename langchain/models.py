import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


def set_api_key():
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def test_get_completion(question: str):
    print(get_completion(question))


def translator_prompt(text: str, style: str):
    template_string = """
    Translate the text that is delimited by triple backticks into a style that is {style}.
    
    text: ```{text}```
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)
    messages = prompt_template.format_messages(style=style, text=text)
    return messages


def simple_review_prompt(text: str):
    review_template = """
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,and output them as a comma separated Python list. 

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """
    prompt_template = ChatPromptTemplate.from_template(template=review_template)
    messages = prompt_template.format_messages(text=text)
    return messages


def generate_response(prompt: str):
    chat_openai = ChatOpenAI(temperature=0)
    response = chat_openai(prompt)
    return response


def getResponseSchema():
    gift_schema = ResponseSchema(
        name="gift",
        description="Was the item purchasedas a gift for someone else? \
            Answer True if yes,False if not or unknown.",
    )
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description="How many days did it take for the productto arrive? If this information is not found, \
            output -1.",
    )
    price_value_schema = ResponseSchema(
        name="price_value",
        description="Extract any sentences about the value or price, and output them as a \
            comma separated Python list.",
    )

    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
    return response_schemas


def formatted_review_prompt(text: str, format_instructions: str):
    
    """
    When we use an f-string to define the review_template string, 
    the values of the text and format_instructions variables are inserted into the string immediately. 
    This means that we don’t need to pass these variables to the format_messages method because their values 
    are already included in the template string.
    """
    
    review_template = """
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.\

    delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.\

    price_value: Extract any sentences about the value or price,and output them as a comma separated Python list.
    

    text: {text}
    

    {format_instructions}
    """


    prompt = ChatPromptTemplate.from_template(template=review_template)
    message = prompt.format_messages(text=text, format_instructions=format_instructions)
    return message


if __name__ == "__main__":
    ##0
    set_api_key()

    ### 1. test get_completion
    simple_questions = ["What is the capital of France", "What is the result of 12 + 8"]
    for q in simple_questions:
        test_get_completion(question=q)

    ### 2. prompts: Add style (for the response)

    email = """
    Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls \
    with smoothie! And to make matters worse,the warranty don't cover the cost of \
    cleaning up me kitchen. I need yer help right now, matey!
    """

    message_style = "American English in a calm and respectful tone"

    prompt = f"""
    Translate the text that is delimited by triple backticks into a style that is {message_style}.
    text: ```{email}```
    """

    print(prompt)
    response = get_completion(prompt)
    print(response)

    ### 2. Chat
    # temperature=0 to make it less random
    chat = ChatOpenAI(temperature=0)

    # Prompt creation and formatting

    customer_messages = translator_prompt(style=message_style, text=email)
    print(customer_messages[0].content)

    service_reply = """
    Hey there customer, the warranty does not cover cleaning expenses for your kitchen 
    because it's your fault that you misused your blender by forgetting to put the lid on before 
    starting the blender. 
    Tough luck! See ya!
    """

    service_style_pirate = "a polite tone that speaks in English Pirate"
    service_messages = translator_prompt(style=service_style_pirate, text=service_reply)

    print(service_messages[0].content)

    ## Using the prompt to call chatAI
    customer_messages_response = generate_response(prompt=customer_messages)
    print(type(customer_messages_response))
    print(customer_messages_response.content)

    service_message_response = generate_response(prompt=service_messages)
    print(service_message_response.content)

    ## More complex prompts

    customer_review = """
        This leaf blower is pretty amazing.  It has four settings:\
        candle blower, gentle breeze, windy city, and tornado. \
        It arrived in two days, just in time for my wife's anniversary present. \
        I think my wife liked it so much she was speechless. \
        So far I've been the only one using it, and I've been \
        using it every other morning to clear the leaves on our lawn. \
        It's slightly more expensive than the other leaf blowers \
        out there, but I think it's worth it for the extra features.
     """

    customer_review_prompt = simple_review_prompt(text=customer_review)
    print(customer_review_prompt[0].content)

    customer_review_response = generate_response(prompt=customer_review_prompt)
    print(type(customer_review_response.content))
    print(customer_review_response.content)

    ## Response parser

    # # Extract information from product review and format the answer into json format
    dict_info = {"gift": False, "delivery_days": 5, "price_value": "pretty affordable!"}

    response_schemas = getResponseSchema()
    print("response_schemas: ", response_schemas)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    print(type(format_instructions))
    print("format_instructions: ", format_instructions)

    formatted_customer_messages = formatted_review_prompt(
        text=customer_review, format_instructions=format_instructions
    )
    print("Formatted content: ")
    print(formatted_customer_messages[0].content)

    print("-" * 10)
    response = generate_response(prompt=formatted_customer_messages)
    print(type(response))  # langchain.schema.AIMessage
    print("Not parsed response: ")
    print(type(response.content))
    print("*"*10)
    print(response.content)

    try:
        output_dict = output_parser.parse(response.content)
        print("Parsed response: ")
        print(output_dict)
        print("*" * 10)
        for k in dict_info:
            print(f"{k}: ", output_dict.get(k))
    except Exception as e:
        print(e)
