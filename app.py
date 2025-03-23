import gradio as gr
from nlp import summarize
from scraping import scrape

def summarize_text(url):
    # Perform text summarization using your desired method
    text = scrape(url)
    normal_summary, gpt_summary, comparison_metrics, ner = summarize(text)
    return normal_summary, gpt_summary, comparison_metrics, ner

# Create a Gradio interface
url_input = gr.components.Textbox(label="Enter URL")
outputs = [
    gr.components.Textbox(label="Normal Text Summary"),
    gr.components.Textbox(label="GPT Text Summary"),
    gr.components.JSON(label="Comparison Metrics for Normal vs GPT Summary"),
    gr.components.HighlightedText(label="Named Entity Recognition")
]


interface = gr.Interface(
    fn=summarize_text,
    inputs=url_input,
    outputs=outputs,
    title="News Summarization and Analysis",
    description="Enter a News Article URL and get your News Summary."
)

# Launch the interface
interface.launch()
