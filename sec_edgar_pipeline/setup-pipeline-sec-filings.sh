#!/usr/bin/env bash
# This script follows setup steps in: https://colab.research.google.com/drive/12mx7QE0Zm4jGB-3yTa9UBRhAsHU0ZScJ
# which is included in: https://github.com/Unstructured-IO/pipeline-sec-filings
# This script is used to setup the pipeline-sec-filings project on a new machine.

# Install pipeline-sec-filings
git clone https://github.com/Unstructured-IO/pipeline-sec-filings.git --depth=1
mv pipeline-sec-filings pipeline_sec_filings
cd pipeline_sec_filings || exit

pipenv install ratelimit unstructured nltk
