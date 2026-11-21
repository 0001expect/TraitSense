from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Return list of requirements from requirements.txt."""
    requirements: List[str] = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="mbti_personality_prediction",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end MBTI personality type prediction from text",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
