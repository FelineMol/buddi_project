from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
 
# Load environment variables from .env file

load_dotenv()
 
def get_chat(model: str = "gpt-4.1", streaming: bool = True) -> AzureChatOpenAI:
    """
    Construct the AzureChatOpenAI client.

    Args:
        model: Azure OpenAI deployment/model name.
        streaming: Enable server streaming and usage streaming.

    Returns:
        AzureChatOpenAI instance configured for this project.
    """
    extra = {"stream_options": {"include_usage": True}} if streaming else None
    return AzureChatOpenAI(
        model=model,
        extra_body=extra,
        streaming=streaming,
        stream_usage=streaming,        
        temperature=0.7,
        # max_tokens=None,
        # timeout=None,
        # max_retries=2,
    )
