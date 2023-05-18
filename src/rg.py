import spacy
import evaluate
from transformers import pipeline
from datasets import load_dataset


nlp_nb = spacy.load("nb_core_news_sm")  # remember to download: python -m spacy download nb_core_news_sm
rouge_score = evaluate.load("rouge")


def correctFormat(summary, max_sentences=None):
    """
    Corrects the format of a summary for evaluating with rouge scores. The correct format requires sentences to be
    seperated by newlines. Source: https://github.com/google-research/google-research/tree/master/rouge

    :param summary: (str) A summary of some text. The summary can have multiple sentences or just a single sentence.
    :param max_sentences: (int) Maximum sentences for each summary, the remaining sentences will be ignored.
    :return: String with each sentence of the summary seperated with a newline.
    """
    doc = nlp_nb(summary)
    sentences = [sent.text for sent in doc.sents]

    if max_sentences is None:
        return "\n".join(sentences)
    else:
        return "\n".join(sentences[:max_sentences])


def correctFormatList(summary_list, max_sentences=None):
    """ See docs for correctFormat. This method does the same but with a list of summaries."""
    summary_list_corrected = []

    for i in range(len(summary_list)):
        summary_list_corrected.append(correctFormat(summary_list[i], max_sentences=max_sentences))

    return summary_list_corrected


def compareRouge(label_summaries, pred_summaries):
    """
    Gives rouge score for two lists of summaries on normal format. Summary lists does not need to be on correct format
    :param label_summaries: Label summaries
    :param pred_summaries: Predicted summaries
    :return: Rouge scores
    """
    label_summaries_corrected = correctFormatList(label_summaries)
    pred_summaries_corrected = correctFormatList(pred_summaries)
    scores = rouge_score.compute(predictions=pred_summaries_corrected, references=label_summaries_corrected)
    return scores


# finner rouge score ved å kun velge lead anntall første setninger
def getLeadRougeScore(dataset, text_name, label_name, lead):
    """
    Calculates the lead-X score for a dataset

    :param dataset: Dataset used to compute lead score. NB! Not a DatasetDict but normally a test dataset
    :param text_name: Name of the column with text being summarized in the dataset
    :param label_name: Name of the column with labels (summaries) in the dataset
    :param lead: Number of sentences in the lead summary, a summary consisting of the first x sentences of some text
    :return: Rouge scores
    """
    lead_summaries = correctFormatList(dataset[text_name], max_sentences=lead)
    scores = rouge_score.compute(predictions=lead_summaries, references=dataset[label_name])
    return scores


def getPredictions(summarizer, dataset, text_name, max_length):
    """
    Calculates summarization based on a summarizer object (from huggingface) for an entire dataset.

    :param summarizer: Summarizer used when summarizing the text of the dataset
    :param dataset: Dataset to summarize. NB! Not a DatasetDict but normally a test dataset
    :param text_name: Name of the column with text being summarized in the dataset
    :return: List of the predictions from the dataset, each entry is a String
    """
    texts = [i[text_name] for i in dataset]
    predictions = summarizer(texts, max_length=max_length)
    predictions = [i["summary_text"] for i in predictions]

    for i in range(len(predictions)):
        row = predictions[i].replace("<extra_id_0>", "")
        predictions[i] = row

    return predictions


def getRougeScore(summarizer, dataset, text_name, label_name, max_length):
    """
    Calculates summarization based on a summarizer object (from huggingface) for an entire dataset.

    :param summarizer: Summarizer used when summarizing the text of the dataset
    :param dataset: Dataset to summarize. NB! Not a DatasetDict but normally a test dataset
    :param text_name: Name of the column with text being summarized in the dataset
    :param label_name: Name of the column with labels (summaries) in the dataset
    :param max_length: Max length of summary in predictions
    :return: Rouge scores
    """
    predictions = getPredictions(summarizer, dataset, text_name, max_length)
    return compareRouge(dataset[label_name], predictions)

