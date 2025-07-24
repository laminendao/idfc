from setuptools import setup, find_packages

setup(
    name='idfc',
    version='0.1.0',
    description='Interpretable Divisive Feature Clustering (IDFC) for explainable dimensionality reduction',
    author='Mouhamadou Lamine Ndao, Genane Youness, Ndeye Niang, Gilbert Saporta',
    author_email='mlndao@cesi.fr',
    url='https://github.com/username/idfc',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3'
    ],
)