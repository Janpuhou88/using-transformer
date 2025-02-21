from transformers import MarianMTModel, MarianTokenizer

def translate_chinese_to_english(text):
    # Load the pre-trained model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")

    # Perform the translation
    translated_tokens = model.generate(**tokenized_text)

    # Decode the translated tokens to a string
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

if __name__ == "__main__":
    # Example Chinese text
    chinese_text = "你好，世界！这是一个测试。"

    # Translate the text
    english_translation = translate_chinese_to_english(chinese_text)

    # Print the result
    print(f"Chinese: {chinese_text}")
    print(f"English: {english_translation}")