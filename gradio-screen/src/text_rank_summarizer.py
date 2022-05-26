import spacy
import pytextrank
from math import sqrt
from operator import itemgetter

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('textrank')


def _phrase_vector(doc):
    phrase_id = 0
    unit_vector = []
    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]

    for p in doc._.phrases:
        unit_vector.append(p.rank)
        for chunk in p.chunks:
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

    sum_ranks = sum(unit_vector)
    return [rank / sum_ranks for rank in unit_vector], sent_bounds


def _sent_rank(unit_vector, sent_bounds):
    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        sum_sq = 0.0
        for phrase_id in range(len(unit_vector)):
            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id] ** 2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1
    return sent_rank


def _rank_to_summary(sent_rank, doc, summary_lines):
    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    summary = []
    num_sent = 0
    for sent_id, _ in sent_rank:
        num_sent += 1
        summary.append(sent_text[sent_id])
        if num_sent == summary_lines:
            break

    return ' '.join(summary)


def summarize(text, summary_lines=4):
    doc = nlp(text)
    phrase_vector, sent_bounds = _phrase_vector(doc)
    sent_rank  = sorted(_sent_rank(phrase_vector, sent_bounds).items(), key=itemgetter(1))
    return _rank_to_summary(sent_rank, doc, summary_lines)
