import gradio as gr
from gradio import inputs
from src.text_rank_summarizer import summarize as summarize_extractive
from src.transformer_summarization import summarize_abstractive

interface_abstractive = gr.Interface(
    fn=summarize_abstractive,
    inputs=[gr.inputs.Textbox(), inputs.Number()],
    outputs=gr.outputs.Textbox(label="Summary by Abstractive Method/Encoder-Decoder"),
)

interface_extractive = gr.Interface(
    fn=summarize_extractive,
    inputs=[gr.inputs.Textbox(), inputs.Number()],
    outputs=gr.outputs.Textbox(label="Summary by Extractive Method/pyTextRank"),
)

interfaces = gr.Parallel(
    interface_abstractive,
    interface_extractive,
    title="Compare 2 Text Summarizers",
    inputs=[
        gr.inputs.Textbox(lines=200, label="Paste some English text here"),
        inputs.Number(default=4, label="Number of summary lines (optional)"),
    ],
    live=False,
)


if __name__ == "__main__":
    app, local_url, share_url = interfaces.launch()
