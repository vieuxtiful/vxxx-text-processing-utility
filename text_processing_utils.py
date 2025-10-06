"""
Text Processing Utilities
=========================

A comprehensive collection of text processing utilities for NLP and text analysis tasks.
Provides common text cleaning, normalization, tokenization, and analysis functions
that are frequently needed in text processing pipelines.

Features:
- Text cleaning and normalization
- Tokenization and sentence splitting
- Language detection and encoding handling
- Text statistics and metrics
- Unicode and encoding utilities
- Pattern matching and extraction
- Text similarity and comparison
- Multilingual support

Author: vieuxtiful
License: MIT
"""

import re
import unicodedata
import string
import logging
from typing import List, Dict, Tuple, Optional, Set, Union, Any, Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import math

# Setup logging
logger = logging.getLogger(__name__)

# Constants
PUNCTUATION = set(string.punctuation)
WHITESPACE = set(string.whitespace)
DIGITS = set(string.digits)

class TextCase(Enum):
    """Text case options"""
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    SENTENCE = "sentence"
    PRESERVE = "preserve"

class NormalizationLevel(Enum):
    """Text normalization levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"

@dataclass
class TextStatistics:
    """Text statistics container"""
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

@dataclass
class CleaningOptions:
    """Text cleaning configuration"""
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    case_handling: TextCase = TextCase.PRESERVE
    remove_punctuation: bool = False
    remove_digits: bool = False
    remove_stop_words: bool = False
    custom_patterns: Optional[List[str]] = None

class TextProcessor:
    """
    Main text processing class with comprehensive text manipulation capabilities
    """
    
    def __init__(self, language: str = "en", stop_words: Optional[Set[str]] = None):
        """
        Initialize text processor
        
        Args:
            language: Primary language for processing
            stop_words: Custom stop words set
        """
        self.language = language
        self.stop_words = stop_words or self._get_default_stop_words(language)
        
        # Compile common regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile frequently used regex patterns"""
        self.patterns = {
            'html': re.compile(r'<[^>]+>'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'whitespace': re.compile(r'\s+'),
            'sentence_boundary': re.compile(r'[.!?]+'),
            'word_boundary': re.compile(r'\b\w+\b'),
            'paragraph': re.compile(r'\n\s*\n'),
            'non_alphanumeric': re.compile(r'[^a-zA-Z0-9\s]'),
            'multiple_punctuation': re.compile(r'[^\w\s]{2,}'),
            'currency': re.compile(r'[$€£¥₹]\s*\d+(?:[.,]\d+)*'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'),
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+'),
            'emoji': re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+')
        }
    
    def _get_default_stop_words(self, language: str) -> Set[str]:
        """Get default stop words for a language"""
        stop_words_dict = {
            'en': {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
                'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
                'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
                'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
                'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
                'come', 'made', 'may', 'part'
            },
            'es': {
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no',
                'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al',
                'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'le', 'ya',
                'o', 'porque', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me',
                'hasta', 'donde', 'quien', 'desde', 'todos', 'durante', 'todo',
                'ella', 'ser', 'han', 'tienen', 'él'
            },
            'fr': {
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
                'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne',
                'se', 'pas', 'tout', 'plus', 'par', 'grand', 'en', 'une', 'être',
                'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une',
                'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand'
            }
        }
        return stop_words_dict.get(language, set())
    
    def clean_text(self, text: str, options: Optional[CleaningOptions] = None) -> str:
        """
        Clean and normalize text according to specified options
        
        Args:
            text: Input text to clean
            options: Cleaning configuration options
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        if options is None:
            options = CleaningOptions()
        
        result = text
        
        # Remove HTML tags
        if options.remove_html:
            result = self.patterns['html'].sub('', result)
        
        # Remove URLs
        if options.remove_urls:
            result = self.patterns['url'].sub('', result)
        
        # Remove email addresses
        if options.remove_emails:
            result = self.patterns['email'].sub('', result)
        
        # Remove phone numbers
        if options.remove_phone_numbers:
            result = self.patterns['phone'].sub('', result)
        
        # Normalize Unicode
        if options.normalize_unicode:
            result = self.normalize_unicode(result)
        
        # Handle case conversion
        if options.case_handling != TextCase.PRESERVE:
            result = self.convert_case(result, options.case_handling)
        
        # Remove punctuation
        if options.remove_punctuation:
            result = self.remove_punctuation(result)
        
        # Remove digits
        if options.remove_digits:
            result = re.sub(r'\d+', '', result)
        
        # Remove stop words
        if options.remove_stop_words:
            result = self.remove_stop_words(result)
        
        # Apply custom patterns
        if options.custom_patterns:
            for pattern in options.custom_patterns:
                result = re.sub(pattern, '', result)
        
        # Remove extra whitespace (usually done last)
        if options.remove_extra_whitespace:
            result = self.patterns['whitespace'].sub(' ', result).strip()
        
        return result
    
    def normalize_unicode(self, text: str, form: str = 'NFKC') -> str:
        """
        Normalize Unicode text
        
        Args:
            text: Input text
            form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(form, text)
    
    def convert_case(self, text: str, case_type: TextCase) -> str:
        """
        Convert text case
        
        Args:
            text: Input text
            case_type: Target case type
            
        Returns:
            Case-converted text
        """
        if case_type == TextCase.LOWER:
            return text.lower()
        elif case_type == TextCase.UPPER:
            return text.upper()
        elif case_type == TextCase.TITLE:
            return text.title()
        elif case_type == TextCase.SENTENCE:
            return text.capitalize()
        else:
            return text
    
    def remove_punctuation(self, text: str, keep_chars: Optional[Set[str]] = None) -> str:
        """
        Remove punctuation from text
        
        Args:
            text: Input text
            keep_chars: Set of punctuation characters to keep
            
        Returns:
            Text without punctuation
        """
        if keep_chars is None:
            keep_chars = set()
        
        return ''.join(char for char in text if char not in PUNCTUATION or char in keep_chars)
    
    def remove_stop_words(self, text: str, custom_stop_words: Optional[Set[str]] = None) -> str:
        """
        Remove stop words from text
        
        Args:
            text: Input text
            custom_stop_words: Additional stop words to remove
            
        Returns:
            Text without stop words
        """
        stop_words = self.stop_words.copy()
        if custom_stop_words:
            stop_words.update(custom_stop_words)
        
        words = self.tokenize_words(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    def tokenize_words(self, text: str, preserve_case: bool = True) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            preserve_case: Whether to preserve original case
            
        Returns:
            List of word tokens
        """
        words = self.patterns['word_boundary'].findall(text)
        return words if preserve_case else [word.lower() for word in words]
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be enhanced with more sophisticated methods
        sentences = self.patterns['sentence_boundary'].split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def extract_patterns(self, text: str, pattern_name: str) -> List[str]:
        """
        Extract specific patterns from text
        
        Args:
            text: Input text
            pattern_name: Name of pattern to extract
            
        Returns:
            List of extracted patterns
        """
        if pattern_name in self.patterns:
            return self.patterns[pattern_name].findall(text)
        return []
    
    def calculate_statistics(self, text: str) -> TextStatistics:
        """
        Calculate comprehensive text statistics
        
        Args:
            text: Input text
            
        Returns:
            TextStatistics object with various metrics
        """
        if not text:
            return TextStatistics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        
        # Basic counts
        character_count = len(text)
        words = self.tokenize_words(text)
        word_count = len(words)
        sentences = self.tokenize_sentences(text)
        sentence_count = len(sentences)
        paragraphs = self.patterns['paragraph'].split(text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        line_count = text.count('\n') + 1
        
        # Unique words
        unique_words = len(set(word.lower() for word in words))
        
        # Average calculations
        average_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0.0
        average_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0
        
        # Simple readability score (Flesch-like)
        readability_score = self._calculate_readability(words, sentences)
        
        return TextStatistics(
            character_count=character_count,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            line_count=line_count,
            unique_words=unique_words,
            average_word_length=average_word_length,
            average_sentence_length=average_sentence_length,
            readability_score=readability_score
        )
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """
        Calculate a simple readability score
        
        Args:
            words: List of words
            sentences: List of sentences
            
        Returns:
            Readability score (0-100, higher is more readable)
        """
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count in a word
        
        Args:
            word: Input word
            
        Returns:
            Estimated syllable count
        """
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def calculate_similarity(self, text1: str, text2: str, method: str = "jaccard") -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method ('jaccard', 'cosine', 'overlap')
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(self.tokenize_words(text1.lower()))
        words2 = set(self.tokenize_words(text2.lower()))
        
        if method == "jaccard":
            return self._jaccard_similarity(words1, words2)
        elif method == "cosine":
            return self._cosine_similarity(text1, text2)
        elif method == "overlap":
            return self._overlap_similarity(words1, words2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity"""
        words1 = self.tokenize_words(text1.lower())
        words2 = self.tokenize_words(text2.lower())
        
        # Create word frequency vectors
        all_words = set(words1 + words2)
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _overlap_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate overlap similarity"""
        intersection = len(set1.intersection(set2))
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0.0
    
    def extract_keywords(self, text: str, max_keywords: int = 10, min_frequency: int = 2) -> List[Tuple[str, int]]:
        """
        Extract keywords from text based on frequency
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            min_frequency: Minimum frequency for a word to be considered
            
        Returns:
            List of (keyword, frequency) tuples
        """
        words = self.tokenize_words(text.lower())
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Filter by minimum frequency and return top keywords
        keywords = [(word, freq) for word, freq in word_freq.items() if freq >= min_frequency]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords[:max_keywords]
    
    def generate_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Generate n-grams from text
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        words = self.tokenize_words(text.lower())
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        Simple language detection based on character patterns
        
        Args:
            text: Input text
            
        Returns:
            Detected language code or None
        """
        # This is a very basic implementation
        # In practice, you'd use a proper language detection library
        
        # Count character frequencies
        char_freq = Counter(text.lower())
        
        # Simple heuristics based on common characters
        if any(char in char_freq for char in 'àáâãäåæçèéêëìíîïñòóôõöøùúûüý'):
            return 'fr'  # French
        elif any(char in char_freq for char in 'ñáéíóúü'):
            return 'es'  # Spanish
        elif any(char in char_freq for char in 'äöüß'):
            return 'de'  # German
        else:
            return 'en'  # Default to English
    
    def calculate_text_hash(self, text: str, algorithm: str = 'sha256') -> str:
        """
        Calculate hash of text content
        
        Args:
            text: Input text
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            Hex digest of text hash
        """
        text_bytes = text.encode('utf-8')
        
        if algorithm == 'md5':
            return hashlib.md5(text_bytes).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(text_bytes).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(text_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

# Convenience functions for quick operations
def clean_text(text: str, **kwargs) -> str:
    """Quick text cleaning function"""
    processor = TextProcessor()
    options = CleaningOptions(**kwargs)
    return processor.clean_text(text, options)

def get_text_statistics(text: str) -> TextStatistics:
    """Quick text statistics function"""
    processor = TextProcessor()
    return processor.calculate_statistics(text)

def tokenize(text: str, level: str = 'words') -> List[str]:
    """Quick tokenization function"""
    processor = TextProcessor()
    if level == 'words':
        return processor.tokenize_words(text)
    elif level == 'sentences':
        return processor.tokenize_sentences(text)
    else:
        raise ValueError("Level must be 'words' or 'sentences'")

def extract_keywords(text: str, max_keywords: int = 10) -> List[Tuple[str, int]]:
    """Quick keyword extraction function"""
    processor = TextProcessor()
    return processor.extract_keywords(text, max_keywords)

def calculate_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
    """Quick similarity calculation function"""
    processor = TextProcessor()
    return processor.calculate_similarity(text1, text2, method)
