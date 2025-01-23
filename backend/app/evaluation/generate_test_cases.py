from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    query: str
    expected_docs: List[Dict[str, Any]]
    query_type: str
    description: str


def generate_test_cases_from_article1() -> List[TestCase]:
    """
    Generate test cases from article1.md content.
    Each test case focuses on different aspects of the content
    and different types of queries.
    """
    # Base content from article1.md
    article1_base = {"source": "article1.md", "title": "Biodeutsch"}

    case1 = TestCase(
        query="Was ist biodeutsch?",
        expected_docs=[
            {
                "content": "Das Schlagwort biodeutsch (auch bio-deutsch) bezeichnet seit den 1990er Jahren ethnische Deutsche.",
                "metadata": article1_base,
            }
        ],
        query_type="definition",
        description="Testing basic term definition retrieval",
    )

    case2 = TestCase(
        query="Wann wurde der Begriff biodeutsch erstmals verwendet?",
        expected_docs=[
            {
                "content": "Zum ersten Mal verwendete der deutsch-türkische Karikaturist Muhsin Omurca die Bezeichnung „Bio-Deutscher“ 1996 in einem Cartoon in der taz",
                "metadata": article1_base,
            }
        ],
        query_type="factual",
        description="Testing origin date retrieval",
    )

    case4 = TestCase(
        query="Wer verbreitete biodeutsch in einem satirischen Kurzfilm?",
        expected_docs=[
            {
                "content": "Das Kölner Netzwerk Kanak Attak popularisierte die Bezeichnungen „bio-deutsch“ und „Bio-Deutsche“ 2002 im satirischen Kurzfilm Weißes Ghetto",
                "metadata": article1_base,
            }
        ],
        query_type="factual",
        description="Testing cultural reference recognition",
    )

    case5 = TestCase(
        query="Warum wurde biodeutsch zum Unwort des Jahres 2024 gewählt?",
        expected_docs=[
            {
                "content": "Als politischer Kampfbegriff behauptet er dort eine angeblich existierende gemeinsame genetisch-biologische Herkunft aller „echten“ Deutschen. Es wurde zum Unwort des Jahres 2024 gewählt.",
                "metadata": article1_base,
            }
        ],
        query_type="context",
        description="Testing sociolinguistic impact analysis",
    )

    case6 = TestCase(
        query="Welcher Politiker verwendete biodeutsch in seinem Buchtitel?",
        expected_docs=[
            {
                "content": "Der iranischstämmige Grünen-Politiker Omid Nouripour verwendet die Bezeichnung in seinem Buch Kleines Lexikon für MiMiMis und Bio-Deutsche (2014) scherzhaft.",
                "metadata": article1_base,
            }
        ],
        query_type="factual",
        description="Testing literary reference identification",
    )

    case7 = TestCase(
        query="Wann wurde biodeutsch in den Duden aufgenommen?",
        expected_docs=[
            {
                "content": "Das Wort wurde 2017 in den Duden aufgenommen",
                "metadata": article1_base,
            }
        ],
        query_type="factual",
        description="Testing lexicographical fact retrieval",
    )

    # Return list of test cases
    return [case1, case2, case4, case5, case6, case7]


def verify_test_cases(db, test_cases: List[TestCase]) -> bool:
    """
    Verify that all test cases reference content that exists in the database.
    """
    for case in test_cases:
        for expected_doc in case.expected_docs:
            # Search for content in database
            results = db.collection.get(
                where={"content": {"$contains": expected_doc["content"][:50]}}
            )
            if not results["documents"]:
                logger.warning(
                    f"Content not found for test case: {case.query}\n"
                    f"Expected content: {expected_doc['content'][:50]}..."
                )
                return False
    return True


def main():
    """Generate and verify test cases."""
    test_cases = generate_test_cases_from_article1()

    # Print test cases in a readable format
    for idx, case in enumerate(test_cases, 1):
        print(f"\nTest Case {idx}:")
        print(f"Query: {case.query}")
        print(f"Type: {case.query_type}")
        print(f"Description: {case.description}")
        print("Expected Content:")
        for doc in case.expected_docs:
            print(f"- {doc['content'][:100]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
