import gradio as gr
from gradio import inputs
# from src.text_rank_summarizer import summarize
from src.transformer_summarization import summarize

long_text_input = inputs.Textbox(lines=200, label='Long Text')
summary_lines = inputs.Number(default=4, label='Summary Lines')

interface = gr.Interface(fn=summarize,
                         inputs=[long_text_input],
                         outputs=['text'],
                         live=False,
                         layout='horizontal',
                         css='css/index.css')

if __name__ == '__main__':
    app, local_url, share_url = interface.launch()
