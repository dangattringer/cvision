import setuptools
from setuptools import find_packages
import re
from pathlib import Path

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")


def get_version():
    file = PARENT / 'cvision/__init__.py'
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding="utf-8"), re.M)[1]
    

setuptools.setup(
    name='cvision',
    version=get_version(),
    author='Daniel Gattringer',
    author_email='daniel.gattringer100@gmail.com',
    description='A set of easy-to-use utilities for any computer vision project.',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python'
    ],
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    keywords="computer-vision, deep-learning, vision, CV, DL, AI",
    python_requires='>=3.7',
)