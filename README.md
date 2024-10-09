![alt text](https://github.com/thanapol2/Mean_EBinning/blob/082cd9447659d9d140acc38d5d4c11db9187d06c/Documents/shizuoka%20bannar.png)


# Dissertation "Research on Time Series Decomposition for Real-time Analysis of Dynamic and Stable Behaviors in Streaming Data"
![SAC23](https://img.shields.io/badge/ACM%20SAC-2023-orange.svg?style=flat-square) [![SCAI2024](https://img.shields.io/badge/SCAI-2024-green.svg?style=flat-square)](https://iaiai.org/conference/aai2024/conferences/scai-2024/) [![ECML24](https://img.shields.io/badge/ECML%20PKDD-2024-blue.svg?style=flat-square)](https://2024.ecmlpkdd.org/)

- Official source code of the dissertation "Research on Time Series Decomposition for Real-time Analysis of Dynamic and Stable Behaviors in Streaming Data" by Thanapol Phungtua-eng at Shizuoka University.
- Source codes were implemented using Python 3.9.2 and R 4.3.0.
- To access links and reproduce synthetic and real datasets, please see the information in our official papers below.
- The real-world light curve dataset was provided by M. Aizawa, K. Kashiyama, and S. Sako, obtained from the Tomo-e Gozen project of the Kiso Schmidt telescope. For more details, visit https://tomoe.mtk.ioa.s.u-tokyo.ac.jp/.
- All URLs of existing methods utilized in this thesis are provided in Appendix B of this thesis.


## Official paper

1.  **Elastic data binning (E-Binning)**
```
@article{10.1145/3610409.3610410,
    author = {Phungtua-eng, Thanapol and Sako, Shigeyuki and Nishikawa, Yushi and Yamamoto, Yoshitaka},
    title = {Elastic Data Binning: Time-Series Sketching for Time-Domain Astrophysics Analysis},
    year = {2023},
    issue_date = {June 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {23},
    number = {2},
    issn = {1559-6915},
    url = {https://doi.org/10.1145/3610409.3610410},
    doi = {10.1145/3610409.3610410},
    abstract = {Time-domain astrophysics analysis (TDAA) involves observational surveys of celestial phenomena that may contain irrelevant information because of several factors, one of which is the sensitivity of the optical telescopes. Data binning is a typical technique for removing inconsistencies and clarifying the main characteristics of the original data in astrophysics analysis. It splits the data sequence into smaller bins with a fixed size and subsequently sketches them into a new representation form. In this study, we introduce a novel approach, called elastic data binning (EBinning), to automatically adjust each bin size using two statistical metrics based on the Student's t-test for linear regression and Hoeffding inequality. EBinning outperforms well-known algorithms in TDAA for extracting relevant characteristics of time-series data, called lightcurve. We demonstrate the successful representation of various characteristics in the lightcurve gathered from the Kiso Schmidt telescope using EBinning and its applicability for transient detection in TDAA.},
    journal = {SIGAPP Appl. Comput. Rev.},
    month = jul,
    pages = {5–22},
    numpages = {18},
    keywords = {time-series sketching, student's t-test, lightcurve, hoeffding inequality, data binning}
}
```
2. **Online Season Length Estimation (OnlineSLE)**
```
@inproceedings{OnlineSLE,
	author = {Phungtua-Eng, T. and Yamamoto, Y.},
	booktitle = {Proceedings of the 16th International Congress on Advanced Applied Informatics},
	note = {(To appear)},
	title = {A Fast Season Length Estimation using Sliding Discrete Fourier Transform for Time Series Streaming Data},
	year = {2024}
}
```
3. **Adaptive Seasonal-Trend Decomposition (ASTD)**
```
@InProceedings{10.1007/978-3-031-70344-7_25,
	author="Phungtua-eng, Thanapol and Yamamoto, Yoshitaka",
	editor="Bifet, Albert and Davis, Jesse and Krilavi{\v{c}}ius, Tomas and Kull, Meelis and Ntoutsi, Eirini and {\v{Z}}liobait{\.{e}}, Indr{\.{e}}",
	title="Adaptive Seasonal-Trend Decomposition for Streaming Time Series Data with Transitions and Fluctuations in Seasonality",
	booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
	year="2024",
	publisher="Springer Nature Switzerland",
	address="Cham",
	pages="426--443",
	isbn="978-3-031-70344-7"
}
```

## Contact
If you have any question, please contact thanapol@yy-lab.info

## Link
- [Our laboraory at Shizuoka University](http://lab.inf.shizuoka.ac.jp/yamamoto/)
- [Elastic Data Binning website](https://sites.google.com/view/elasticdatabinning/)
- [Online Season Length Estimation](https://github.com/thanapol2/onlineSLE)
- [Adaptive Seasonal-Trend Decomposition website](https://sites.google.com/view/astd-ecmlpkdd/)