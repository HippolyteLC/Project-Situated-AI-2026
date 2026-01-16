# Project-Situated-AI-2026
Group repo for PSAI 2026

### INSTRUCTIONS
1. The extended_data.[ttl/ nt/ owl] files contain our extended ontology.
2. The psai1_student_notebook_colab_or_local_v2 contains our final notebook and main implementation. All helper functions mentioned in the paper are available in this notebook.
3. Other notebooks contain prior versions of our pipeline.
4. Results have been stored in JSON files in the final_responses folder.
5. Notes.txt file contains some notes for our own use.
6. Examples.txt contain the provided examples and our extension with the 10 dialogues and the matching queries.

### IMPORTANT
- The psaiv2 notebook can be run both locally and on Google Colab, for kaggle use, the directories and imports/ installs need slight modification.
- In the code cells and markdown cells at the start of the notebook some comments are added as to what needs to be done to swap to Colab or local use. Most importantly, is the colab = False/ True variable, and, when running ollama in colab, the timeout + Popen server cells might need to be run repeatedly. When interrupting ollama generations, you need to rerun the ollama run command + timeout & Popen cells. 
- PIPs: owlready2, owlapy, ollama, rdflib, parsimonious, sparqlwrapper. 
