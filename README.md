# RAMOO2024
Bi-objective branch and bound algorithm for 0-1 integer linear programming problems. Developed as a "toy" for the RAMOO workshop 2024 in Wuppertal.

# The problems tackled
The code solves bi-objective linear 0-1 problems of the form
$$
\begin{align}
\min\ & Cx\\
\text{s.t.}\ & Ax\geq b,\\
&x\in\{0,1\}^n
\end{align}
$$
