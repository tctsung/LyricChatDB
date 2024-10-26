# set working directory to LyricChat repo root
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
os.chdir(repo_root)

# Add repo root to Python's path
sys.path.append(repo_root)

# self-written functions:
import src.LLM as LLM
from src.PROMPT import SystemPrompt
from src.LLM import human_msg, AI_msg, sys_msg, lyric_msg
from src.utils import jsonl_append, read_jsonl

# structured output:
from pydantic import BaseModel, Field, model_validator, field_validator
from enum import Enum
from typing import Literal

# basics:
from tqdm import tqdm
import json


# emotion labels to classify
class Emotions(str, Enum):
    Joy = "Joy"
    Love = "Love"
    Nostalgia = "Nostalgia"
    Sadness = "Sadness"
    Anger = "Anger"
    Fear = "Fear"
    Hope = "Hope"
    Desire = "Desire"
    Confidence = "Confidence"
    Regret = "Regret"
    Peace = "Peace"
    Excitement = "Excitement"
    Loneliness = "Loneliness"
    Gratitude = "Gratitude"
    Confusion = "Confusion"
    Betrayal = "Betrayal"
    Ambition = "Ambition"
    Forgiveness = "Forgiveness"
    Freedom = "Freedom"


# summary_description
SUMMARY_DESCRIPTION = """
A 100-200 word paragraph summarizing the lyrics directly, capturing key messages, core emotional themes, and feelings. 
Focus on describing the life situations and experiences portrayed, without using phrases like 'the song' or 'the lyrics.'
"""


# Schema for structured LLM output
class LyricExtract(BaseModel):
    """
    TODO: Schema for Instructor structured LLM output
    """

    summary: str = Field(..., description=SUMMARY_DESCRIPTION)
    emotions: list[Emotions] = Field(
        description="The top two emotions observed in the lyrics. Two emotions must be different."
    )

    @field_validator("summary")
    def summary_is_50_words(cls, val):
        # check if summary is long enough
        if len(val.split()) < 50:
            raise ValueError("Current summary to short, it must be more than 80 words")
        return val

    @field_validator("emotions")
    def Emotions_are_diff(cls, val):
        # check if primary_emotion and secondary_emotion are different
        if val[0] == val[1]:
            raise ValueError("primary_emotion and supporting_emotion must be different")
        return [c.value for c in val]  # turn ENUM to str after validation


def main():
    start_idx = 4
    # read lyrics:
    input_path = "data/NEFFEX_2024_09_19_23_07_06/lyrics_processed.json"
    output_path = "data/NEFFEX_2024_09_19_23_07_06/summary_and_emotions.json"
    with open(input_path, "r") as f:
        lyrics = json.load(f)
    # load model (make sure ollama server is running):
    llm = LLM.InstructorLLM("local")
    # generate prompt:
    msg = [sys_msg(SystemPrompt.practical), None]  # None is buffer for input lyrics
    res_lst = []  # buffer for model output
    i = 0
    for key, lyric in tqdm(lyrics.items()):
        if i < start_idx:
            i += 1
            continue
        msg[1] = lyric_msg(lyric)  # iterate through lyrics
        res = llm.run(
            msg, schema=LyricExtract, max_retries=10
        )  # LLM response summary & classified emotions

        # append each output to local:
        primary_emotion, supporting_emotion, summary = (
            res.emotions[0],
            res.emotions[1],
            res.summary,
        )
        dct = {
            "key": key,
            "primary_emotion": primary_emotion,
            "supporting_emotion": supporting_emotion,
            "summary": summary,
        }
        res_lst.append(dct)
        jsonl_append(output_path, dct)


if __name__ == "__main__":
    main()  # the func to run
