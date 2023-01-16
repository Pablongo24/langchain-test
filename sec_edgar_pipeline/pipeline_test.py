"""
Based on: https://colab.research.google.com/drive/12mx7QE0Zm4jGB-3yTa9UBRhAsHU0ZScJ
"""
import pickle
import nltk
from sec_edgar_pipeline.pipeline_sec_filings.prepline_sec_filings.fetch import get_form_by_ticker, open_form_by_ticker
from unstructured.documents.html import HTMLDocument
from unstructured.nlp.partition import is_possible_title, is_possible_narrative_text
from sec_edgar_pipeline.pipeline_sec_filings.prepline_sec_filings.sections import (
    section_string_to_enum, validate_section_names, SECSection
)
from sec_edgar_pipeline.pipeline_sec_filings.prepline_sec_filings.sec_document import (
    SECDocument, REPORT_TYPES, VALID_FILING_TYPES
)
from unstructured.documents.elements import Title
from unstructured.cleaners.core import clean_extra_whitespace
from unstructured.staging.label_studio import stage_for_label_studio
import re

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")


def is_10k_item_title(title: str) -> bool:
    """Determines if a title corresponds to a 10-K item heading."""
    return ITEM_TITLE_RE.match(title) is not None


if __name__ == "__main__":
    text = get_form_by_ticker(
        'rgld', '10-K', company='TeXT eXploration Technologies', email='pablo@textexp.tech'
    )

    html_doc = HTMLDocument.from_string(text).doc_after_cleaners(skip_headers_and_footers=True, skip_table_text=True)

    ITEM_TITLE_RE = re.compile(
        r"(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?\.?:?"
    )

    for element in html_doc.elements:
        element.text = clean_extra_whitespace(element.text)
        if isinstance(element, Title) and is_10k_item_title(element.text):
            print(element)

    sec_doc = SECDocument.from_string(text)
    risk_narrative = sec_doc.get_section_narrative(SECSection.RISK_FACTORS)
    compiled_data = stage_for_label_studio(elements=risk_narrative)
    with open('data.pickle', 'wb') as f:
        pickle.dump(compiled_data, f)
