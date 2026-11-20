---
orphan: true
---

# 2PLM-RANK

These supplementary notes explain the successive-choice response law and
shared-statement constraints. The [2PLM-RANK example](../tutorials/two_pl_rank_workflow.ipynb)
fits the model to observed HEXACO triplets and shows the fitted statement
endorsement curves.

## Response probabilities

For respondent p and statement j, define log endorsement odds

$$
\ell_{pj}=q_j a_j\theta_{p,t(j)}-\beta_j.
$$

Here `loading_magnitude` is a, the design keying q gives its sign, and
`statement_intercept` is beta. Increasing beta lowers statement utility. Beta is already
an intercept; it is not a difficulty subsequently multiplied by a loading.
For a complete preference order r of a K-statement block,

$$
P(r)=\prod_{h=1}^{K-1}
\frac{\exp(\ell_{p,r_h})}
{\sum_{s=h}^{K}\exp(\ell_{p,r_s})}.
$$

PICK and MOLE probabilities sum over compatible complete orders. For two
statements this reduces to the logistic probability of the utility difference.
For larger blocks the sequential law differs from MGPPM's score-weighted
ranking-pattern softmax.

## Reference

Zheng, C., Liu, J., Li, Y., Xu, P., Zhang, B., Wei, R., Zhang, W., Liu, B., & Huang, J. (2024). A 2PLM-RANK multidimensional forced-choice model and its fast estimation algorithm. *Behavior Research Methods, 56*(6), 6363–6388. <https://doi.org/10.3758/s13428-023-02315-x>

The response law follows its successive-choice construction.
