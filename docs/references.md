# Dataset and Method References

## Datasets

1. **Wisconsin Diagnostic Breast Cancer.** UCI Machine Learning Repository, dataset 17. The local `wdbc.data` file has 569 observations and 30 predictive features after dropping the ID column. UCI dataset page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic. Original publication listed by UCI: W. H. Wolberg, O. L. Mangasarian, and N. Street, “Nuclear feature extraction for breast tumor diagnosis,” *Electronic Imaging*, 1993. Dataset DOI: https://doi.org/10.24432/C5DW2B.

2. **Pima Indians Diabetes.** The local 768-row, 8-feature file is the commonly used Pima Indians Diabetes binary classification dataset, not the unrelated UCI “Diabetes” time-series dataset. Cite J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, “Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,” *Proceedings of the Symposium on Computer Applications in Medical Care*, 1988, pp. 261–265. Dataset description and citation context: https://search.r-project.org/CRAN/refmans/reglogit/html/pima.html. The exact provenance of the specific `Diabetes.csv` copy in this repository should be recorded (download URL/version/hash) before publication.

3. **Cleveland Heart Disease.** UCI Machine Learning Repository, Heart Disease dataset 45, Cleveland processed file. The local file has 303 records and 13 predictors; the original target values 0–4 are binarized to 0 versus 1–4. UCI dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease. Dataset DOI: https://doi.org/10.24432/C52P4G. Original paper listed by UCI: R. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” *American Journal of Cardiology*, 1989.

## Methods to cite in the manuscript

4. **Support Vector Machines.** C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, 20, 273–297 (1995). https://doi.org/10.1007/BF00994018.

5. **Particle Swarm Optimization.** J. Kennedy and R. Eberhart, “Particle swarm optimization,” *Proceedings of ICNN'95 – International Conference on Neural Networks*, 1995, pp. 1942–1948. https://doi.org/10.1109/ICNN.1995.488968.

6. **Nested cross-validation / model selection.** M. Stone, “Cross-validatory choice and assessment of statistical predictions,” *Journal of the Royal Statistical Society: Series B*, 36(2), 111–147 (1974). https://doi.org/10.1111/j.2517-6161.1974.tb00994.x.

7. **Scikit-learn implementation.** F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, 12, 2825–2830 (2011). https://jmlr.org/papers/v12/pedregosa11a.html.

## Citation caveat
The project should preserve the exact download URL and a checksum for each raw dataset file. Dataset pages establish the dataset identity, but do not independently verify the provenance of the exact local file copies.
