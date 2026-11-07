"""
Text Normalization for Informal and Code-Mixed Healthcare Communication.
Handles:
- Elongated characters (e.g. 'feveeeer' -> 'fever', 'bohooot' -> 'bohot')
- Punctuation spacing and cleaning
- Noise and URL removal
- Whitespace regularization
"""

import re
import unicodedata

class TextNormalizer:
    def __init__(self):
        # Common medical abbreviations used by Indian patients/practitioners
        self.abbreviations = {
            r'\bbp\b': 'blood pressure',
            r'\bhr\b': 'heart rate',
            r'\bdr\b': 'doctor',
            r'\byrs\b': 'years',
            r'\byr\b': 'year',
            r'\bmins\b': 'minutes',
            r'\bhrs\b': 'hours',
            r'\btab\b': 'tablet',
            r'\binj\b': 'injection',
            r'\brx\b': 'prescription'
        }

    def reduce_elongated_characters(self, text: str) -> str:
        """
        Compresses repeated characters occurring 3 or more times down to 2 characters,
        e.g., 'painnnnn' -> 'painn', 'bohooot' -> 'bohot'.
        """
        # Replace 3 or more repeated characters with 1 or 2
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def normalize_spacing(self, text: str) -> str:
        """Ensures punctuation doesn't stick directly to words without proper spacing."""
        text = re.sub(r'([.,!?;:()\[\]])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean(self, text: str, expand_abbreviations: bool = False) -> str:
        """
        Full cleaning pipeline for patient text.
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs and email mentions
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Handle elongated characters
        text = self.reduce_elongated_characters(text)

        if expand_abbreviations:
            for pattern, replacement in self.abbreviations.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

if __name__ == "__main__":
    normalizer = TextNormalizer()
    sample = "Doctorrrr saab, mujhe bohoooot severe feveeer ho raha hai... BP is 150/90 since 2 days!!!"
    print("Original :", sample)
    print("Cleaned  :", normalizer.clean(sample))
