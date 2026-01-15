# /// script
# requires-python = "==3.12"
# dependencies = [
#     "altair==6.0.0",
#     "polars==1.37.1",
#     "requests==2.32.5",
#     "tenacity==9.1.2",
#     "vl-convert-python==1.9.0",
# ]
# ///
import argparse
import os

from argparse import Namespace
from typing import Any

import altair as alt
import polars as pl
import requests

from altair import Chart, TitleParams
from polars import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential

parser = argparse.ArgumentParser(
    prog="graph", description="Generate graphs from a database"
)
parser.add_argument("chart_type", type=str)
args: Namespace = parser.parse_args()


def calculate_metrics(
    df: DataFrame, col: str, ans_col: str, count_col: str, pct_col: str
) -> DataFrame:
    """Organize a column from a DataFrame to get the count of the answers and
    their percentages.

    Args:
    df (DataFrame): The DataFrame to get the column from.
    col (str): The column in the DataFrame.
    ans_col (str): The name to be given to the column containing the answers.
    count_col (str): The name to be given to the column containing the count.
    pct_col (str): The name to be given to the column containing the percentages.
    """

    return (
        df.select(pl.col(col).str.strip_chars(".").alias(ans_col))
        .group_by(ans_col)
        .len(name=count_col)
        .with_columns((pl.col(count_col) / pl.col(count_col).sum()).alias(pct_col))
        .sort(count_col, descending=True)
    )


class Question:
    """Simple class for wrapping a DataFrame column to make it better for
    generating charts.

    Attributes:
        value (DataFrame): The column in DataFrame format in a 'answers-count'
        pair, to be passed to a chart as data.
        answers (str): The label of the column containing the unique answers.
        counts (str): The label of the column containing each answer's count.
        question (str): The title of the column, which is the original question
        that was asked.
    """

    def __init__(self, col: str, df: DataFrame):
        self.question: str = col
        self.answers: str = "Respostas"
        self.counts: str = "Contagem"
        self.percent: str = "Porcentagem"
        self.value: DataFrame = calculate_metrics(
            df,
            self.question,
            ans_col=self.answers,
            count_col=self.counts,
            pct_col=self.percent,
        )


def gen_pie_chart(question: Question, filename: str, dir: str = ".") -> None:
    """Generate a pie chart in PDF format of a question.

    Args:
    question (Question): The question to generate the chart from.
    filename (str): The name the file with the question's chart will get.
    dir (str): The directory the generated file will be placed. Defaults to the
    current directory.
    """

    base: Chart = alt.Chart(question.value, title=question.question).encode(
        theta=alt.Theta(question.counts, stack=True)
    )
    pie = base.mark_arc(outerRadius=120).encode(
        color=alt.Color(
            question.answers,
            legend=alt.Legend(labelLimit=500, title="Legenda", orient="right"),
        ),
        order=alt.Order(question.counts, sort="descending"),
    )
    text = base.mark_text(radius=140).encode(
        text=alt.Text(question.percent, format=".1%"),
        order=alt.Order(question.counts, sort="descending"),
        color=alt.value("black"),
    )
    chart = pie + text

    os.makedirs(dir, exist_ok=True)
    chart.save(f"{dir}/{filename}.pdf")


def gen_bar_chart(question: Question, filename: str, dir: str = ".") -> None:
    """Generate a bar chart in PDF format of a question.

    Args:
    question (Question): The question to generate the chart from.
    filename (str): The name the file with the question's chart will get.
    dir (str): The directory the generated file will be placed. Defaults to the
    current directory.
    """

    total = question.value.select(pl.col(question.counts).sum()).item()
    base: Chart = alt.Chart(question.value, title=question.question).encode(
        x=alt.X(question.counts, scale=alt.Scale(domain=[0, total])),
        y=alt.Y(
            question.answers,
            sort="-x",  # Descending order.
            axis=alt.Axis(title=None, labelLimit=400),
        ),
    )
    pie = base.mark_bar().encode(
        color=alt.Color(question.answers, legend=None),
    )
    text = base.mark_text(align="left", baseline="middle", dx=3).encode(
        text=alt.Text(question.percent, format=".1%"), color=alt.value("black")
    )
    chart = pie + text

    os.makedirs(dir, exist_ok=True)
    chart.save(f"{dir}/{filename}.pdf")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data(url: str, filename: str) -> None:
    """Downloads the data file if it's not in the current directory already.

    Args:
    url (str): The URL to download from.
    filename (str): Local path to save the file.
    """

    if not os.path.exists(filename):
        print(f"{parser.prog}: File {filename} not found. Downloading...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)


def main() -> None:
    url: str = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTOHDQBlfxZ9wKL5_80fPcM5uJcm6ftUSBSi1y9pvIONMtygAw_YtWWNWIdxZvndRy-0W-sU1dH3dLf/pub?gid=1668449075&single=true&output=csv"
    get_data(url, "data.csv")
    df: DataFrame = pl.read_csv("data.csv")
    chart_types: dict[str, Any] = {
        "pie": gen_pie_chart,
        "bar": gen_bar_chart,
    }
    questions: list[tuple[Question, str]] = [
        (
            Question(
                "Com que frequência você utiliza ferramentas de Inteligência Artificial (como ChatGPT, Gemini, Copilot) para auxiliar nas tarefas da disciplina de programação?",
                df,
            ),
            "frequency",
        ),
        (Question("Quais ferramentas você utiliza?", df), "tools"),
        (
            Question(
                "Em qual etapa do desenvolvimento do código você sente maior necessidade de usar a IA?",
                df,
            ),
            "step",
        ),
        (
            Question(
                "Quando a IA gera um código para você, o que você costuma fazer?", df
            ),
            "action",
        ),
        (
            Question(
                "Você sente que o uso da IA atrapalha a sua capacidade de encontrar erros (debugar) sozinho?",
                df,
            ),
            "debugging",
        ),
        (
            Question(
                "Se você tivesse que fazer uma prova prática hoje, sem acesso à internet ou IA, como avaliaria sua confiança para resolver os problemas?",
                df,
            ),
            "test",
        ),
        (
            Question(
                "Você acredita que aprende menos quando utiliza a IA para gerar a resposta de um exercício?",
                df,
            ),
            "learning",
        ),
        (
            Question(
                "Na sua opinião, o uso de IA em disciplinas introdutórias deveria ser",
                df,
            ),
            "opinion",
        ),
        (
            Question(
                "Para você, haveria algum problema em admitir ao seu professor que usou IA para realizar uma tarefa?",
                df,
            ),
            "professor",
        ),
    ]

    for question in questions:
        chart_types[args.chart_type](question[0], question[1], "build")
        print(f"{parser.prog}: generated {args.chart_type} chart for {question[1]}")


if __name__ == "__main__":
    main()
