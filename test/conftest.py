import pytest
from transformers import MarkupLMProcessor


@pytest.fixture
def test_html():
    # The HTML from MarkupLM paper
    return  """
<html>
<head>
    <title>Galaxy S20</title>
</head>
<body>
    <div>
        <li>
            <div>
                <span> Display </span>
                <span> 6.5 inch </span>
            </div>
        </li>
        <li>
            <div>
                <span> Processor </span>
                <span> Qualcomm Snapdragon </span>
            </div>
        </li>
        <ul>
            <li>
                <span> Release Date </span>
                2020
            </li>
        </ul>
    </div>
</body>
</html>
"""

@pytest.fixture(scope="session")
def markuplm_processor():
    processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    yield processor
    processor.parse_html = True

@pytest.fixture(scope="session")
def id2xpath_tag(markuplm_processor):
    id2xpath_tag = {v: k for k, v in markuplm_processor.tokenizer.tags_dict.items()}
    id2xpath_tag[len(id2xpath_tag)] = "<UNK>"
    id2xpath_tag[len(id2xpath_tag)] = "<PAD>"
    yield id2xpath_tag
