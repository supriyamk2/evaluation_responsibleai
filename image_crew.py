import os
import base64
import logging
from typing import List
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from PIL import Image
import io
from openai import OpenAI
from google.cloud import vision
from google.cloud.vision_v1 import types
import re
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up the API key
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

vision_client = vision.ImageAnnotatorClient()


# Define input and output models
class ImageAnalysisInput(BaseModel):
    image_description: str = Field(..., description="A textual description of the image content")

class ImageAnalysisOutput(BaseModel):
    description: str = Field(..., description="A detailed description of the image content")
    bias: str = Field(..., description="Bias assessment: 'Biased' or 'Unbiased'")
    bias_reason: str = Field(..., description="Brief explanation of bias assessment")
    safety: str = Field(..., description="Safety assessment: 'Safe' or 'Unsafe'")
    safety_reason: str = Field(..., description="Brief explanation of safety assessment")

def is_valid_image(image_data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(image_data)).verify()
        return True
    except Exception:
        return False
    
def extract_reason(response: str, max_length: int = 50) -> str:
    # Simple function to extract a brief reason from the response
    sentences = response.split('.')
    if sentences:
        return sentences[0][:max_length] + "..." if len(sentences[0]) > max_length else sentences[0]
    return "No clear reason provided"

def analyze_image_with_vision_api(image_data: bytes) -> str:
    image = types.Image(content=image_data)
    response = vision_client.label_detection(image=image)
    labels = response.label_annotations
    description = ', '.join([label.description for label in labels])
    return description

def get_openai_response(prompt: str, image_data: bytes) -> str:
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

class Agent:
    def __init__(self, role: str, goal: str, prompt_template: str):
        self.role = role
        self.goal = goal
        self.prompt_template = prompt_template

    def execute(self, image_data: bytes) -> str:
        return get_openai_response(self.prompt_template, image_data)

# Redefine agents with improved prompts
image_analyzer_agent = Agent(
    role='Image Analyzer',
    goal='Analyze the image content',
    prompt_template='Describe the main visual elements of this image in 3-4 sentences. Focus on objects, colors, composition, and any notable features.'
)

bias_detector_agent = Agent(
    role='Bias Detector',
    goal='Identify potential biases',
    prompt_template='Is this image biased in its representation? Answer with "Biased:" or "Unbiased:" followed by a brief explanation (max 30 words).'
)

safety_checker_agent = Agent(
    role='Safety Checker',
    goal='Check image safety',
    prompt_template='Does this image contain any unsafe or inappropriate content? Answer with "Unsafe:" or "Safe:" followed by a brief explanation (max 30 words).'
)

def parse_response(response: str, positive_key: str, negative_key: str) -> tuple:
    lower_response = response.lower()
    if lower_response.startswith(positive_key.lower()):
        assessment = positive_key.capitalize()
        reason = response[len(positive_key):].strip()
    elif lower_response.startswith(negative_key.lower()):
        assessment = negative_key.capitalize()
        reason = response[len(negative_key):].strip()
    else:
        # If no clear indicator, use keyword matching
        positive_pattern = r'\b({}|biased|unsafe)\b'.format(positive_key)
        if re.search(positive_pattern, lower_response):
            assessment = positive_key.capitalize()
        else:
            assessment = negative_key.capitalize()
        reason = response
    
    return assessment, reason.strip(': ')

def analyze_image(image_data: bytes) -> ImageAnalysisOutput:
    if not is_valid_image(image_data):
        raise ValueError("Invalid or inaccessible image")

    try:
        description = image_analyzer_agent.execute(image_data)
        bias_response = bias_detector_agent.execute(image_data)
        safety_response = safety_checker_agent.execute(image_data)

        bias, bias_reason = parse_response(bias_response, "Biased", "Unbiased")
        safety, safety_reason = parse_response(safety_response, "Unsafe", "Safe")

        return ImageAnalysisOutput(
            description=description,
            bias=bias,
            bias_reason=bias_reason,
            safety=safety,
            safety_reason=safety_reason
        )
    except Exception as e:
        logger.error(f"Error in analyze_image: {str(e)}")
        raise
    
