# LingPro NLP

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/text-processing-utils.svg)](https://badge.fury.io/py/text-processing-utils)

A comprehensive collection of text processing utilities for NLP and text analysis tasks. This library provides common text cleaning, normalization, tokenization, and analysis functions that are frequently needed in text processing pipelines.

## Features

- **Text Cleaning**: Remove HTML, URLs, emails, phone numbers, and custom patterns
- **Normalization**: Unicode normalization, case conversion, whitespace handling
- **Tokenization**: Word and sentence tokenization with customizable options
- **Statistics**: Comprehensive text metrics including readability scores
- **Language Detection**: Basic language identification capabilities
- **Similarity Calculation**: Multiple similarity metrics (Jaccard, Cosine, Overlap)
- **Keyword Extraction**: Frequency-based keyword identification
- **N-gram Generation**: Generate n-grams for text analysis
- **Pattern Extraction**: Extract emails, URLs, hashtags, mentions, and more
- **Multilingual Support**: Built-in support for multiple languages

## Installation

```bash
pip install text-processing-utils
```

For development dependencies:
```bash
pip install text-processing-utils[dev]
```

## Quick Start

### Basic Text Cleaning

```python
from text_processing_utils import TextProcessor, CleaningOptions

# Initialize processor
processor = TextProcessor(language='en')

# Basic cleaning
text = "Visit https://example.com for more info! Email: contact@example.com"
cleaned = processor.clean_text(text)
print(cleaned)  # "Visit  for more info! Email: "

# Custom cleaning options
options = CleaningOptions(
    remove_urls=True,
    remove_emails=True,
    remove_extra_whitespace=True,
    case_handling=TextCase.LOWER
)
cleaned = processor.clean_text(text, options)
```

### Text Statistics

```python
from text_processing_utils import get_text_statistics

text = """
This is a sample text for analysis. It contains multiple sentences.
The text processing utility will analyze various metrics.
"""

stats = get_text_statistics(text)
print(f"Words: {stats.word_count}")
print(f"Sentences: {stats.sentence_count}")
print(f"Readability: {stats.readability_score}")
print(f"Average word length: {stats.average_word_length}")
```

### Tokenization

```python
from text_processing_utils import tokenize

text = "Natural language processing is fascinating!"

# Word tokenization
words = tokenize(text, level='words')
print(words)  # ['Natural', 'language', 'processing', 'is', 'fascinating']

# Sentence tokenization
sentences = tokenize("Hello world! How are you? I'm fine.", level='sentences')
print(sentences)  # ['Hello world', ' How are you', " I'm fine"]
```

### Keyword Extraction

```python
from text_processing_utils import extract_keywords

text = """
Machine learning is a subset of artificial intelligence. 
Machine learning algorithms build models based on training data.
Artificial intelligence and machine learning are transforming technology.
"""

keywords = extract_keywords(text, max_keywords=5)
print(keywords)  # [('machine', 3), ('learning', 3), ('artificial', 2), ...]
```

### Text Similarity

```python
from text_processing_utils import calculate_similarity

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown fox leaps over a lazy dog"

# Jaccard similarity
jaccard = calculate_similarity(text1, text2, method='jaccard')
print(f"Jaccard similarity: {jaccard:.3f}")

# Cosine similarity
cosine = calculate_similarity(text1, text2, method='cosine')
print(f"Cosine similarity: {cosine:.3f}")
```

## Advanced Usage

### Custom Text Processor

```python
from text_processing_utils import TextProcessor, CleaningOptions, TextCase

# Initialize with custom stop words
custom_stop_words = {'the', 'and', 'or', 'but', 'custom', 'words'}
processor = TextProcessor(language='en', stop_words=custom_stop_words)

# Advanced cleaning configuration
options = CleaningOptions(
    remove_html=True,
    remove_urls=True,
    remove_emails=True,
    normalize_unicode=True,
    case_handling=TextCase.LOWER,
    remove_stop_words=True,
    custom_patterns=[r'\d+', r'[^\w\s]']  # Remove digits and punctuation
)

text = "<p>Visit https://example.com! Contact: test@email.com (123) 456-7890</p>"
cleaned = processor.clean_text(text, options)
print(cleaned)
```

### Pattern Extraction

```python
processor = TextProcessor()

text = """
Contact us at support@company.com or visit https://company.com
Call us at (555) 123-4567 or follow @company on social media
Check out #innovation and #technology trends
"""

# Extract different patterns
emails = processor.extract_patterns(text, 'email')
urls = processor.extract_patterns(text, 'url')
phones = processor.extract_patterns(text, 'phone')
hashtags = processor.extract_patterns(text, 'hashtag')
mentions = processor.extract_patterns(text, 'mention')

print(f"Emails: {emails}")
print(f"URLs: {urls}")
print(f"Phones: {phones}")
print(f"Hashtags: {hashtags}")
print(f"Mentions: {mentions}")
```

### N-gram Analysis

```python
processor = TextProcessor()

text = "Natural language processing enables computers to understand human language"

# Generate bigrams
bigrams = processor.generate_ngrams(text, n=2)
print("Bigrams:", bigrams)

# Generate trigrams
trigrams = processor.generate_ngrams(text, n=3)
print("Trigrams:", trigrams)
```

### Language Detection

```python
processor = TextProcessor()

texts = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "Hallo, wie geht es dir?"
]

for text in texts:
    language = processor.detect_language(text)
    print(f"'{text}' -> {language}")
```

### Text Hashing

```python
processor = TextProcessor()

text = "This is some important content that needs to be hashed"

# Different hash algorithms
md5_hash = processor.calculate_text_hash(text, 'md5')
sha256_hash = processor.calculate_text_hash(text, 'sha256')

print(f"MD5: {md5_hash}")
print(f"SHA256: {sha256_hash}")
```

## API Reference

### TextProcessor Class

The main class for text processing operations.

#### Constructor
```python
TextProcessor(language='en', stop_words=None)
```

#### Methods

- **`clean_text(text, options=None)`**: Clean and normalize text
- **`tokenize_words(text, preserve_case=True)`**: Tokenize into words
- **`tokenize_sentences(text)`**: Split into sentences
- **`calculate_statistics(text)`**: Get comprehensive text statistics
- **`extract_patterns(text, pattern_name)`**: Extract specific patterns
- **`calculate_similarity(text1, text2, method='jaccard')`**: Calculate text similarity
- **`extract_keywords(text, max_keywords=10, min_frequency=2)`**: Extract keywords
- **`generate_ngrams(text, n=2)`**: Generate n-grams
- **`detect_language(text)`**: Detect text language
- **`calculate_text_hash(text, algorithm='sha256')`**: Calculate text hash

### CleaningOptions Class

Configuration for text cleaning operations.

```python
CleaningOptions(
    remove_html=True,
    remove_urls=True,
    remove_emails=True,
    remove_phone_numbers=True,
    remove_extra_whitespace=True,
    normalize_unicode=True,
    case_handling=TextCase.PRESERVE,
    remove_punctuation=False,
    remove_digits=False,
    remove_stop_words=False,
    custom_patterns=None
)
```

### TextStatistics Class

Container for text analysis results.

```python
@dataclass
class TextStatistics:
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    line_count: int
    unique_words: int
    average_word_length: float
    average_sentence_length: float
    readability_score: float
    language_detected: Optional[str] = None
    encoding_detected: Optional[str] = None
```

### Convenience Functions

Quick access functions for common operations:

- **`clean_text(text, **kwargs)`**: Quick text cleaning
- **`get_text_statistics(text)`**: Quick statistics calculation
- **`tokenize(text, level='words')`**: Quick tokenization
- **`extract_keywords(text, max_keywords=10)`**: Quick keyword extraction
- **`calculate_similarity(text1, text2, method='jaccard')`**: Quick similarity calculation

## Supported Patterns

The library can extract the following patterns:

- **HTML tags**: `<tag>content</tag>`
- **URLs**: `http://example.com`, `https://example.com`
- **Email addresses**: `user@domain.com`
- **Phone numbers**: `(555) 123-4567`, `555-123-4567`
- **Hashtags**: `#hashtag`
- **Mentions**: `@username`
- **Currency**: `$100`, `€50`, `£25`
- **Dates**: `12/31/2023`, `31-12-2023`
- **Times**: `14:30`, `2:30 PM`
- **Emojis**: Unicode emoji characters

## Language Support

Built-in stop words and language-specific processing for:

- English (en)
- Spanish (es)
- French (fr)
- German (de) - basic support
- Italian (it) - basic support
- Portuguese (pt) - basic support

## Performance Considerations

- Regex patterns are compiled once during initialization for efficiency
- Large texts are processed in chunks where possible
- Memory usage is optimized for typical text processing workflows
- Consider using generators for very large datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v1.0.0
- Initial release
- Basic text cleaning and normalization
- Tokenization and statistics
- Pattern extraction
- Similarity calculation
- Keyword extraction
- N-gram generation
- Language detection
- Comprehensive documentation

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/lexiq-team/text-processing-utils/issues) on GitHub.
