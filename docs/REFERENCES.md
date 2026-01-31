# OpenAD-lib References

## Formatted Citations

1. Dekhici B. Data-driven modeling, order reduction and control of anaerobic digestion processes. Ph.D. thesis, University of Tlemcen (2024)

2. Dekhici B, Benyahia B, Cherki B, Fiori L, Andreottola G. Modeling of biogas production from hydrothermal carbonization products in a continuous anaerobic digester. ACM Trans Model Comput Simul 34: (2024) https://doi.org/10.1145/3680281

3. Batstone D, Keller J, Angelidaki I, Kalyuzhnyi S, Pavlostathis S, Rozzi A, Sanders W, Siegrist H, Vavilin V. The IWA Anaerobic Digestion Model No 1 (ADM1). Water Sci Technol 45:65-73 (2002) https://doi.org/10.2166/wst.2002.0292

4. Bernard O, Hadj-Sadok Z, Dochain D, Genovesi A, Steyer J-P. Dynamical model development and parameter identification for an anaerobic wastewater treatment process. Biotechnol Bioeng 75:424-438 (2001) https://doi.org/10.1002/bit.10036

5. Dekhici B, Short M. Data-Driven Modelling of Biogas Production Using Multi-Task Gaussian Processes. Systems and Control Transactions 4:26-32 (2025) https://doi.org/10.69997/sct.121877

6. Murali R, Bywater A, Dolat M, Dekhici B, Zarei M, Hilton L, Sadhukhan J, Zhang D, Short M. Anaerobic Digestion Site-Wide Optimisation and Decision-Making: An Industrial Perspective and Review. Renew Sustain Energy Rev 226:116402 (2026)

7. Arnell M, Astals S, Åmand L, Batstone DJ, Jensen PD, Jeppsson U. Modelling anaerobic co-digestion in Benchmark Simulation Model No. 2: Parameter estimation, substrate characterisation and plant-wide integration. Water Res 98:138-146 (2016) https://doi.org/10.1016/j.watres.2016.03.070

8. Rosen C, Jeppsson U. Aspects on ADM1 implementation within the BSM2 framework. Department of Industrial Electrical Engineering and Automation, Lund University (2006) https://www.iea.lth.se/publications/Reports/LTH-IEA-7224.pdf

9. Sadrimajd P, Mannion P, Howley E, Lens PNL. PyADM1: a Python implementation of Anaerobic Digestion Model No. 1. bioRxiv (2021) https://doi.org/10.1101/2021.03.03.433746

10. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework. arXiv:1907.10902 (2019) https://arxiv.org/abs/1907.10902

11. Murali R, Dekhici B, Chen T, Zhang D, Short M. Mechanistic and Data-Driven Models for Predicting Biogas Production in Anaerobic Digestion Processes. Systems and Control Transactions 4:388-393 (2025) https://doi.org/10.69997/sct.176459

12. Fiedler F, Karg B, Lüken L, Brandner D, Heinlein M, Brabender F, Lucia S. do-mpc: Towards FAIR nonlinear and robust model predictive control. Control Engineering Practice 140:105676 (2023)

13. OpenAD-lib GitHub repository. https://github.com/benmola/OpenAD-lib

14. Optuna GitHub repository. https://github.com/optuna/optuna

15. Supergen Bioenergy Impact Hub. Rapid digitalisation of bioenergy for higher efficiency and profit. https://www.supergen-bioenergy.net/research/digitalisation-and-ai-for-ad/

16. Sadrimajd P, Mannion P, Howley E, Lens PNL. PyADM1: A Python implementation of Anaerobic Digestion Model No. 1. bioRxiv (2021) https://doi.org/10.1101/2021.03.03.433746

17. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A next-generation hyperparameter optimization framework. arXiv:1907.10902 (2019) https://arxiv.org/abs/1907.10902

---

## BibTeX Entries

### PyADM1
```bibtex
@article{Sadrimajd2021,
    author = {Sadrimajd, Peyman and Mannion, Patrick and Howley, Enda and Lens, Piet N. L.},
    title = {PyADM1: a Python implementation of Anaerobic Digestion Model No. 1},
    year = {2021},
    doi = {10.1101/2021.03.03.433746},
    journal = {bioRxiv}
}
```

### Optuna
```bibtex
@misc{Akiba2019,
    title={Optuna: A Next-generation Hyperparameter Optimization Framework}, 
    author={Takuya Akiba and Shotaro Sano and Toshihiko Yanase and Takeru Ohta and Masanori Koyama},
    year={2019},
    eprint={1907.10902},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/1907.10902}
}
```

---

## Categorized References

### OpenAD-lib Core Publications
- [1] Dekhici B. Ph.D. thesis (2024)
- [2] Dekhici et al. ACM TOMACS (2024) - AM2 Model & HTC
- [5] Dekhici & Short. SCT (2025) - MTGP for AD
- [11] Murali, Dekhici et al. SCT (2025) - Mechanistic vs ML Models
- [13] OpenAD-lib GitHub

### Foundational AD Models
- [3] Batstone et al. (2002) - ADM1 Original
- [4] Bernard et al. (2001) - AM2 Model
- [7] Arnell et al. (2016) - BSM2 ACoD
- [8] Rosen & Jeppsson (2006) - ADM1 in BSM2

### Optimization & Control
- [10] Akiba et al. (2019) - Optuna Framework
- [12] Fiedler et al. (2023) - do-mpc
- [14] Optuna GitHub

### Related Work & Reviews
- [6] Murali, Dekhici et al. (2026) - AD Optimization Review
- [9] Sadrimajd et al. (2021) - PyADM1
- [15] Supergen Bioenergy Hub

---

## Usage in Documentation

### Citing in Notebooks
```markdown
**References:**
- **AM2 Model**: [Dekhici et al. (2024)](https://doi.org/10.1145/3680281)
- **MTGP**: [Dekhici & Short (2025)](https://doi.org/10.69997/sct.121877)
- **ADM1**: [Batstone et al. (2002)](https://doi.org/10.2166/wst.2002.0292)
- **Optuna**: [Akiba et al. (2019)](https://arxiv.org/abs/1907.10902)
```

### Citing in Papers
Use the formatted citations above (items 1-15) in your reference list.

### Citing in Code Comments
```python
# Implementation based on:
# Dekhici et al. (2024) - AM2 model calibration
# Akiba et al. (2019) - Optuna optimization
```
