import json
import os
from datetime import datetime
from typing import Union

import pandas as pd
from dotenv import load_dotenv
from langchain import OpenAI, Prompt
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from pandas import DataFrame

load_dotenv()


def get_summary(input_text):
    llm = OpenAI(temperature=0)
    prompt_template = """Write a concise summary of the following:
    
    {text}
    
    CONCISE SUMMARY:
    """

    prompt = Prompt(template=prompt_template, input_variables=["text"])
    text_splitter = CharacterTextSplitter()
    mr_chain = MapReduceChain.from_params(llm=llm, prompt=prompt, text_splitter=text_splitter)
    return mr_chain.run(input_text)


def timestamp_to_utc(timestamp: float | int) -> datetime:
    return datetime.utcfromtimestamp(timestamp)


def utc_to_timestamp(utc_time: datetime) -> int:
    return int(utc_time.timestamp())


def extract_text(json_data: Union[dict, str, list]) -> str:
    try:
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
    except json.decoder.JSONDecodeError:
        pass
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == "text":
                yield value
            elif isinstance(value, (dict, list)):
                yield from extract_text(value)
    elif isinstance(json_data, list):
        for item in json_data:
            yield from extract_text(item)


def filter_dataframe(
        df: DataFrame,
        user: Union[str, list, None] = None,
        time_range: Union[tuple[int | float, int | float], None] = None,
        channel: Union[str, list, None] = None
) -> DataFrame:
    if user:
        if isinstance(user, str):
            df = df[df["user"] == user]
        elif isinstance(user, list):
            df = df[df["user"].isin(user)]

    if time_range:
        df = df[(df["timestamp"] > time_range[0]) & (df["timestamp"] < time_range[1])]

    if channel:
        if isinstance(channel, str):
            df = df[df["channel"] == channel]
        elif isinstance(channel, list):
            df = df[df["channel"].isin(channel)]

    return df


def join_text_from_df(df: DataFrame) -> str:
    df = df[df["text"].notna()]
    text_ = df["text"].tolist()
    return '\n\n'.join(text_)


if __name__ == '__main__':
    CSV_FILENAME = 'slack_logs.csv'
    PARENT_DIR = os.path.join('..')
    DATA_DIR = os.path.join(PARENT_DIR, 'data')

    df_ = pd.read_csv(os.path.join(DATA_DIR, CSV_FILENAME))
    prelim_data = filter_dataframe(
        df_,
        time_range=(utc_to_timestamp(datetime(2023, 1, 8)), utc_to_timestamp(datetime(2023, 1, 16))),
        channel="chat-gpt"
    )
    text = join_text_from_df(prelim_data)
    summary = get_summary(text)
    print(summary)
