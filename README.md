# Counterfactual Bounds for Algorithmic Recourse 

The original $\textbf{SCM}$ is reformulated to $\SCM(\Mcal^{R})$ using the response function framework (....Sec) as  
$\Mcal^{\texttt{R}}=(\Xb, \Rb, \mb, \PP_\Rb)$, where $\Rb=\{R_1, ..., R_n\}$ are response function variables 
% (i.e., $R_i$ corresponds to the equivalence classes of $U_i$ w.r.t.\ $f_i$) 
with (unknown) joint distribution $\PP_\Rb$ \footnote{$\PP_\Rb$ follows the same known factorisation as $\PP_\Ub$, i.e., dependence between any 
($U_i, U_j)$ is reflected in a dependence between $(R_i, R_j)$.}, and $\mb$ are the re-parameterized structural equations of the form
\begin{align*}
    \begin{split}
    \{X_i:=m_i(\pa_i,R_i)\}_{i=1}^{n}, \quad \quad \PP_{\Rb} = P(R_1,.....R_n)
    \end{split}
\end{align*}
Additionally, it is assumed that we have access to a probabilistic classifier $h:\Xcal_1\times ...\times \Xcal_n\rightarrow\Ycal\subseteq[0,1]$ which is used in a consequential decision making setting\footnote{For example, we can consider the task of loan approval where $h$ is used to decide whether an individual $\xF$ is given a loan (if $h(\xF)\geq0.5$) or not (if $h(\xF)<0.5$).}. We illustrate this situation in Figure 3(b).

\paragraph{Goal} Under these (weak) assumption settings, given an individual $\xF=(x_1^{\texttt{F}},..., x_n^{\texttt{F}})$ which obtained an unfavourable decision, $h(\xF)<0.5$, the approach aims to find the \textit{informative} bounds on the actions in the form of atomic interventions $do(\Xb_\Ical:=\thetab)$ (or $do(\thetaI)$ for short) on a subset of variables $\Xb_\Ical\subseteq \Xb$ with $\Ical\subseteq\{1,..,n\}$ which lead to a more favourable decision, e.g., $\EE[h(\Xb)]>0.5$ where the expectation is taken over the counterfactual (or interventional) distribution of $\Xb$ under $do(\thetaI)$ given $\xF$. \textbf{The bounds suggests whether the recourse is guaranteed or not i.e., whether intervention will flip the decision or not.}
