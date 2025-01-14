from youtube_search import YoutubeSearch  # pip install youtube-search
import os
import sys
import pandas as pd
import uuid  # pip install uuid

# set working directory to LyricChat repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
os.chdir(repo_root)

# Add repo root to Python's path
sys.path.append(repo_root)

# self-written functions:
import src.LLM as LLM
from src.PROMPT import SystemPrompt
from src.LLM import human_msg, AI_msg, sys_msg, lyric_msg
import src.utils as utils

# structured output:
from pydantic import BaseModel, Field, model_validator, field_validator
from enum import Enum
from typing import Literal

# basics:
from tqdm import tqdm
import json


def main():
    # set args:
    input_path = r"data\NEFFEX_2024_09_19_23_07_06\lyrics_processed.json"
    deployment = "local"  # make sure Ollama is activated if use local
    # feature extraction:
    data_prep = DataPrep(input_path, deployment)
    data_prep.save()  # save as data_for_DB.parquet at same folder as input path


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


class DataPrep:
    def __init__(self, input_path, deployment: Literal["cloud", "local"] = "local"):
        """
        TODO: turn scraped lyrics from json format to pd.DF with LLM extracted features (summary, emotions)
        Args:
            input_path: json file generated from src/webscraping
            deployment: model type (cloud- Gemini, local- llama)
        Output (pd.DF):
            table with following features:
            - uuid: row index
            - artist: artist name
            - title: song title
            - youtube_link: youtube link for the song
            - primary_emotion: LLM generated classification
            - supporting_emotion: LLM generated classification
            - summary: LLM generated summary
            - lyric: song lyrics

        """
        # load input:
        self.input_path = input_path
        self.dir_path = os.path.dirname(self.input_path)
        with open(input_path, "r") as f:
            self.lyrics = json.load(f)
        # load model (make sure ollama server is running):
        self.model = LLM.InstructorLLM(deployment)

        # feature extraction:
        self.get_summary_and_emotions()

    def get_summary_and_emotions(self):
        msg = [sys_msg(SystemPrompt.practical), None]  # None is buffer for input lyrics
        res_lst = []  # buffer for model output
        temp_output_path = os.path.join(self.dir_path, "LLM_temp_res.pickle")
        # iterate through all songs:
        for key, lyric in tqdm(self.lyrics.items()):
            # get singer info:
            key_lst = key.split("|||")
            artist, title = key_lst[0], key_lst[1]
            # feature extraction using LLM:
            msg[1] = lyric_msg(lyric)  # update input w current lyrics
            model_output = self.model.run(msg, schema=LyricExtract, max_retries=20)
            res = {
                "artist": artist,
                "title": title,
                "youtube_link": video_link(f"{artist} - {title} lyrics"),
                "primary_emotion": model_output.emotions[0],
                "supporting_emotion": model_output.emotions[1],
                "summary": model_output.summary,
                "lyric": lyric,
            }
            # save current outputs as pickle file to avoid lost all info:
            res_lst.append(res)
            utils.pickle_save(res_lst, temp_output_path)
        # turn list to DF:
        self.df = pd.DataFrame(res_lst)
        self.df.index = [uuid.uuid4().hex for _ in range(len(res_lst))]

    def save(self, output_name="data_for_DB.parquet"):
        # TODO: save df to the same dir as input_path
        output_path = os.path.join(self.dir_path, output_name)
        self.df.to_parquet(output_path)


##### Helper functions ######
def video_link(query):
    results = YoutubeSearch(query, max_results=1).to_dict()
    return f"https://www.youtube.com{results[0]['url_suffix']}"


##### Helper functions ######

if __name__ == "__main__":
    main()
