from transformers import LongformerTokenizer, EncoderDecoderModel

# Load model and tokenizer
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


def summarize(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    # Get the summary from the output tokens
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
