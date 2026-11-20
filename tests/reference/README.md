# Frozen numerical references

These files contain generated numerical results, not third-party implementation
source. Ordinary tests read them without installing R or running MCMC.

- `mfcblockinfo_information.json`: expected probit comparison information from
  actual unmodified MFCblockInfo functions at a pinned source commit. Covers
  two-, three- and four-statement designs and several ability points.
- `mfcblockinfo_joint_information.json`: actual joint Gaussian full-ranking
  probabilities and expected block information from the same reference. Its
  numerical Hessians require different tolerances from analytic pair information;
  the original values, including numerical roundoff, are retained.
- `forcechoice_mggum_information.json`: actual ForceChoice MGGUM probabilities,
  with Fisher information independently derived by numerical differentiation.
  This is not an exported ForceChoice information function. Two derivative step
  settings check numerical stability. The reference's small distance perturbation
  is documented rather than added to the production model.
- `forcechoice_single_item.json`: actual ForceChoice single-item likelihoods.
  These verify likelihood values and are not information reference results.
- `mirt_information.json`: actual `mirt::iteminfo` full matrices and response
  probabilities for fixed 1--4PL, MGPCM and an equivalent binary 2PL ranking
  contrast. The fixed-slope 1PL uses a 2PL item with slope one, not a separately
  fitted Rasch model.
- `rank2pl_author_information.json`: actual author `iteminfo_n` outputs from
  the supplied Rank-2PL workspace, including a numerical-derivative dyad and
  a symbolic-derivative triplet. Five-trait matrices retain inactive directions.
- `mgppm_author_information.json`: actual author triplet/tetrad probabilities
  and symbolic first derivatives, combined into expected Fisher matrices.
- `forcechoice_rank_information.json`: actual ForceChoice FCMIRT/FCGGUM
  probabilities with independently differentiated Fisher matrices, covering
  sizes 2--4 and RANK/PICK/MOLE. This is not the Fu/Joo information implementation.
  The FCGGUM helper's distance perturbation and its public fitting constraints
  are separate from the native GGUM-RANK parameterization.

Each fixture records its inputs, external source identity, parameter conventions,
runtime and output provenance. A reference update must preserve that distinction
and explain any change in numerical tolerances. Never regenerate expected arrays
using the production function being tested. Comparing matrices requires the same
response unit, category law, parameter scale and nuisance conditioning.

- `gtum_stan_likelihood.json`: actual CmdStan 2.40.0 fixed-parameter outputs
  from the original author GTUM binary/graded functions and complete exported
  declarations. All 378 function-category points and 126 whole-declaration points
  retain their inputs, source/tool/CSV hashes and category mappings. This is a
  new reference execution, rather than historical Stan output or a posterior
  comparison. It follows the [GTUM article](https://doi.org/10.1177/10944281231210481).
  Native tests compare every probability, all graded log values and binary log
  values with reference probabilities in `[1e-6, 1-1e-6]`. Full strict-log failure
  diagnostics remain separate: the original binary source uses subtractive
  complements that can lose tail-log precision. A stable scalar logistic-interval
  equation supplies an additional independent log-value check. The two cases in
  `test_gtum.py` record diagnostic counts and any strict-log failures in JUnit;
  ordinary tests neither install nor execute Stan.

External conditional information checks live in `test_information_references.py`.
Native analytic/derivative/missingness checks remain in `test_information.py`;
the standalone joint Gaussian ranking calculation has `test_tirt_information.py`.

- `betareg_information.json`: original pinned betareg expected mean-coefficient
  information from unmodified R fitfun/hessfun expressions evaluated at fixed
  coefficients. Covers logit/probit links and three fixed precision values;
  retains per-response and summed matrices. Precision is conditioned on, not
  eliminated through a nuisance-parameter inverse. No optimization or MCMC.
