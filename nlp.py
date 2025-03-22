import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import openai
import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModel
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import json
import config


# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
openai.api_key = config.API_KEY

# load stopwords and punctuations
stopwords = list(STOP_WORDS)
punctuation = punctuation + "\n"

# load the english spaCy model
nlp = spacy.load("en_core_web_sm")


def summarize(text):
    # tokenize the text
    doc = nlp(text)

    # print("*"*100)
    # print(doc)
    """
    Word Tokenization
    """
    tokens = []
    for token in doc:
        tokens.append(token.text)
    

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
    
    # print("="*100)
    # print("Word Frequencies:", json.dumps(word_frequencies, indent=2))
    # print("="*100)

    max_freq = max(word_frequencies.values())

    for key in word_frequencies.keys():
        word_frequencies[key] /= max_freq

    # print("="*100)
    # print("Word Frequencies Percentage:", json.dumps(word_frequencies, indent=2))
    # print("="*100)

    """
    Sentence Tokenization
    """
    tokenized_sentences = []
    for sentence in doc.sents:
        tokenized_sentences.append(sentence)

    sentence_scores = {}
    for sentence in tokenized_sentences:
        for word in sentence:
            if word.text.lower() in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sentence] += word_frequencies[word.text.lower()]

    # print("sentence_scores:", sentence_scores)
    select = int(len(tokenized_sentences) * 0.3)
    # print("select:", select)

    summary = nlargest(select, sentence_scores, key=sentence_scores.get)

    NER = spacy.load("en_core_web_sm")
    ner_text = NER(text)

    final_summary = [word.text for word in summary]
    length_of_summary = 0
    for i in range(len(final_summary)):
        length_of_summary += len([word for word in final_summary[i].split()])
    summary = " ".join(final_summary)
    gpt_summary = get_news_summary(text, length_of_summary)


    # print("="*100)
    # print("Named Entity Recognition")
    ner = list()
    for word in ner_text.ents:
        ner.append(word.text)
        # print(word.text,word.label_)
    # print("="*100)

    pos_tokens = []

    for token in ner_text:
        if token.ent_type_ != "":
            pos_tokens.extend([(token.text, token.ent_type_), (" ", None)])

    # print("="*100)
    # print("SUMMARY")
    # print(summary)
    # print("="*100)

    # print("="*100)
    # print("GPT - Summary")
    # print(gpt_summary)
    # print("="*100)

    metrics = compare_gpt_normal(gpt_summary, final_summary, ner, text)
    # print("="*100)
    # print("Summary Comparison: ")
    # print("Percentage of NER words retained in GPT Summary: ", gpt_coverage)
    # print("Percentage of NER words retained in Normal Summary: ", normal_coverage)
    # print("Cosine Similarity of both Summary: ", cosine_sim)
    # print("Normal Rouge Score: \n", normal_r_score)
    # print("GPT Rouge Score: \n", gpt_r_score)
    # print("BLEU Normal Summary Score: ", normal_bleu)
    # print("BLEU GPT Summary Score: ", gpt_bleu)
    # print("="*100)

    return summary, gpt_summary, metrics, pos_tokens

def compare_gpt_normal(gpt_summary, normal_summary, ner, text):
    ner = [word.lower() for word in ner]
    normal_summary = " ".join(map(str, normal_summary))

    # Covert to lower case
    gpt_summary = " ".join(map(str, [word.lower() for word in gpt_summary.split()]))
    normal_summary = " ".join(map(str, [word.lower() for word in normal_summary.split()]))
    # Calculate the percentage of words detected by NER (Original Text) present in the summary.
    gpt_coverage = len([word for word in ner if word in gpt_summary])/len(ner)
    normal_coverage = len([word for word in ner if word in normal_summary])/len(ner)
    
    # Encode two sentences
    texts = [gpt_summary, normal_summary]
    inputs = tokenizer.batch_encode_plus(texts, return_tensors = 'pt', padding=True)
    
    # Get the BERT embeddings of the two sentences
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = outputs.last_hidden_state[:, 0, :]

    # Calculate cosine similarity between the two embeddings
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    normal_rouge_scores = scorer.score(text, normal_summary)
    gpt_rouge_score = scorer.score(text, gpt_summary)

    reference = text.split()
    normal_candidate = normal_summary.split()
    gpt_candidate = gpt_summary.split()
    normal_bleu = sentence_bleu(reference, normal_candidate)
    gpt_bleu = sentence_bleu(reference, gpt_candidate)


    return {
            "Normal NER Coverage:": normal_coverage, 
            "GPT-3 NER Coverage:": gpt_coverage, 
            "Cosine Similarity:": cosine_sim.item(), 
            "Normal Summary ROUGE Score:": normal_rouge_scores, 
            "GPT-3 Summary ROUGE Score:": gpt_rouge_score, 
            "Normal Summary BLEU:": normal_bleu, 
            "GPT-3 Summary BLEU:": gpt_bleu
            }


def get_news_summary(text, num_words):
    response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "user", "content": f"{text}; Summarize this text in {num_words} words"}
                                    ]
                                )
    
    summary = response.choices[0]["message"]["content"].strip(".")
    return summary


# Testing 
if __name__ == "__main__":
    text = """
    US Secretary of State Antony Blinken met briefly in New Delhi on Thursday with Russian Foreign Minister Sergei Lavrov, pressing him on Ukraine in the two countries' highest-level one-on-one contact since the war.

    Blinken met for less than 10 minutes with Lavrov on the sidelines of the Group of 20 talks, a day after saying that he had no plans to meet his Russian or Chinese counterparts, a senior US official said.

    Blinken wanted to "disabuse the Russians of any notion that our support might be wavering" on Ukraine, the official said, after growing support from European allies for peace initiatives.

    Blinken wanted to "send that message directly" and also urged Russia to engage with Ukraine on the basis of demands put forward by President Volodymyr Zelensky.

    "We always remain hopeful that the Russians will reverse their decision and be prepared to engage in a diplomatic process that can lead to a just and durable peace," the official said.

    "But I wouldn't say that coming out of this encounter there was any expectation that things would change in the near term."

    Blinken also urged Russia to free Paul Whelan, a former US Marine detained since late 2018, and to reverse President Vladimir Putin's recent decision to suspend the New START nuclear treaty, the last arms control agreement between the Cold War-era foes.

    Blinken told Lavrov that "the treaty is in the interest of both our countries as well as international security, as the world expects us to behave responsibly when it comes to nuclear security," the official said.
    """

    summarize(text)
