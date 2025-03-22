# News Summarization

This project aims to provide an efficient way to summarize news articles using two different methods: Extractive News Summarization and Abstractive News Summarization using GPT-3. It also includes a comparison between the two summary approaches. The application can be used through a web interface created using Gradio.

## Extractive News Summarization

Extractive News Summarization is a technique that involves selecting the most important sentences or phrases from the original news article and combining them to create a summary. This method aims to extract the most relevant information directly from the article without generating new sentences.

## Abstractive News Summarisation using GPT-3

Abstractive News Summarization, on the other hand, employs GPT-3, a powerful language model, to generate summaries that are not limited to extracting sentences from the original article. Instead, it generates new sentences that capture the essence of the news in a more human-like manner.

## Comparison of Summaries

To provide insights into the differences between the two summarisation methods, this project includes a comparison feature. It highlights the variations in the content, style, and length of the extractive and abstractive summaries. This allows users to evaluate and assess which approach aligns better with their requirements.

## Setup Development Environment

### Cloning the Repository

To clone the repository, follow these steps:

- Open your terminal or command prompt.
- Change the current working directory to the location where you want to clone the repository.
- Copy the clone link from the repository.
- Run the following command:

```bash
git clone https://github.com/rutujajangle/News_Summarizer.git
```

> This will clone the repository to your local machine.

### Creating a Virtual Environment

Before installing the project dependencies, it is recommended to create a virtual environment for the project. This will ensure that the project dependencies are isolated from your global Python installation and prevent version conflicts.

- To create a virtual environment, follow these steps:
- Open your terminal or command prompt.
- Change the current working directory to the cloned repository directory.
- Run the following command:

```bash
python3 -m venv News_Summarizer
```

> This will create a virtual environment named `News_Summarizer` in the repository directory.

### Activating the Virtual Environment

- After creating the virtual environment, activate it by running the following command:

```bash
source News_Summarizer/bin/activate
```

> This will activate the virtual environment and you will see (summarize) added to the beginning of your command prompt.

### Installing the Dependencies

- After activating the virtual environment, you can install the project dependencies by running the following command:

```bash
pip install -r requirements.txt
```

> This will install all the required packages listed in the requirements.txt file.

### Deactivating the Virtual Environment

- To deactivate the virtual environment, simply run the following command:

```bash
deactivate
```

> This will deactivate the virtual environment and return you to your global Python installation.

### Running the Summarize App

- To run the Summarize app interface, simply run the following command. This will open up a Web Interface for the app on your loacalhost

```bash
python3 app.py
```
