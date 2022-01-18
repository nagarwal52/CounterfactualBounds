# Counterfactual Bounds for Algorithmic Recourse 

### Motivation
Many algorithms involving high stake decisions ought to provide algorithmic recourse, i.e., why an algorithm made such a decision and/or what changes can be done on the feature(s) side to change the decision made by the algorithm. Though, the standard recourse approaches provides an elegant mathematical framework for performing algorithmic recourse, they are inherently flawed -- either they do not assume the causal dependence in the features or make strong assumptions about the structure like dependence on the data-generation process. In our framework, we aim to highlight these flaws and provide a solution. 

Our work addressed these flaws and models the causal relationships of the features with just the assumption that we know the causal graph and the observational data. Our model provides the bounds on the effects of the actions, which are in the form of hypothetical interventions. 

### Proposed Solution (Theory)
In the work, we argued that algorithmic recourse is inherently a causal problem. Motivated from the seminal work of Balke and Pearl (1994) and Karimi et al (2020), broadly speaking, we proposed an algorithm which aims to find the counterfactual bounds for every actions and actions are performed on the features in the form of atomic interventions. To be specific, given the deterministic classifier (h), observed distribution data (D, here data is discrete), the causal graph (G) and the confounding structure (either full or partial), the algorithm gives the bounds all the elements of the set of actionable nodes. The bounds can either be informative or uninformative, for instance, if Lower bound (LB) > 0.5, the intervention will "flip" (change) the decision or in other words the recourse is gauranteed.   


In causal settings, our approach aims to find the *informative* bounds for every action (in the form of atomic interventions) and suggests whether recourse will 'flip' the decision of an individual

### Proposed Solution (Implementation)

1. To extract the information from the input entities, run causalgraph.py
2. To build the causal graph with response function variables, run responsefunctions.py
3. 
