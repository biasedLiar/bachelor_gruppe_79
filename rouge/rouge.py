import spacy
import evaluate
from datasets import load_dataset
from transformers import pipeline
import random

nlp_nb = spacy.load("nb_core_news_sm")
rouge_score = evaluate.load("rouge")

def correctFormat(summary, max_sentences=None):
    """
    Corrects the format of a summary for evaluating with rouge scores. The correct format requires sentences to be
    seperated by newlines.

    :param summary: String containing a summary of multiple sentences or just a single sentence.
    :param max_sentences: Maximum sentences for each summary, the remaining sentences will be ignored.
    :return: String with each sentence of the summary seperated with a newline.
    """
    doc = nlp_nb(summary)
    sentences = [sent.text for sent in doc.sents]

    if max_sentences is None:
        return "\n".join(sentences)
    else:
        return "\n".join(sentences[:max_sentences])


def correctFormatList(summary_list, max_sentences=None):
    """ See docs for correctFormat. This method does the same but with a list."""
    summary_list_corrected = []

    for i in range(len(summary_list)):
        summary_list_corrected.append(correctFormat(summary_list[i], max_sentences=max_sentences))

    return summary_list_corrected


def compareRouge(label_summaries, pred_summaries):
    """
    Gives rouge score for two lists of summaries on normal format (no newlines).
    :param label_summaries: Label summaries
    :param pred_summaries: Predicted summaries
    :return:
    """
    label_summaries_corrected = correctFormatList(label_summaries)
    pred_summaries_corrected = correctFormatList(pred_summaries)
    scores = rouge_score.compute(predictions=pred_summaries_corrected, references=label_summaries_corrected)
    return scores


# finner rouge score ved å kun velge lead anntall første setninger
def getLeadRougeScore(dataset, lead):
    lead_summaries = correctFormatList(dataset["text"], max_sentences=lead)
    scores = rouge_score.compute(predictions=lead_summaries, references=dataset["label"])
    return scores


def getPredictions(summarizer, dataset):
    predictions = []
    for i in range(len(dataset)):
        predictions += summarizer(dataset[i]["text"])[0]["summary_text"]
    return predictions

def getRougeScore(summarizer, dataset):
    predictions = getPredictions(summarizer, dataset)
    return compareRouge(dataset["label"], predictions)
