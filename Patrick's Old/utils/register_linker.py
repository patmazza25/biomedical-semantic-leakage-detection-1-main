# utils/register_linker.py
from spacy.language import Language

try:
    from scispacy.linking import EntityLinker

    @Language.factory("scispacy_linker")
    def create_scispacy_linker(nlp, name, resolve_abbreviations=True, linker_name="umls"):
        """Factory for UMLS EntityLinker (scispaCy)"""
        return EntityLinker(resolve_abbreviations=resolve_abbreviations, name=linker_name)

    print("✅ Registered 'scispacy_linker' factory.")

except ImportError:
    print("⚠️ SciSpaCy not installed, linker factory not available.")
