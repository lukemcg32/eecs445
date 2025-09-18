1.a.i. 
Giving the ICUTypes increasing values might tell the model the ICUTypes are ordered or spaced out numerically which will cause bias in the system. A better alterative would be to have each ICUType be a separate variable and have each be either True or False (or 0 or 1 for binary represenation).

1.a.ii.
Because we only consider the maximum value, we cannot capture how the patients' condition has changed throughout their visit. Additionally, because we only take the maximum, we have no visibility into low numbers, which could be equally as concerning (i.e. if a patients' temperature dropped significantly). To capture more information, taking the minimum, maximum, and mean would be more helpful.


1.b
This imputation approach assumes that values are missing at random. Another approach could be using the median instead in order to capture more of a trend in the data. However, computing the median is far less efficient.

1.c. 
Without feature normalization regularization unproportionally penalizes features on smaller scales and shrinks their influence relative to features on larger scales. With normalization, we ensure all features are on a "level playing field" and allow weights to be penalized more precisely.

1.d
        Feature  Mean Value  Interquartile Range
            Age      0.6471               0.3378
         Gender      0.5790               1.0000
         Height      0.3722               0.0008
        ICUType      0.5933               0.6667
         Weight      0.2950               0.1136
        max_ALP      0.0717               0.0165
        max_ALT      0.0156               0.0137
        max_AST      0.0157               0.0138
    max_Albumin      0.5053               0.0000
        max_BUN      0.1319               0.0979
  max_Bilirubin      0.0430               0.0252
max_Cholesterol      0.3261               0.0000
 max_Creatinine      0.0649               0.0334
    max_DiasABP      0.3330               0.0394
       max_FiO2      0.7504               0.4167
        max_GCS      0.8772               0.2500
    max_Glucose      0.1389               0.0684
       max_HCO3      0.3503               0.1389
        max_HCT      0.3335               0.1459
         max_HR      0.3294               0.1835
          max_K      0.2094               0.1014
    max_Lactate      0.1360               0.0498
        max_MAP      0.3996               0.0582
         max_Mg      0.0415               0.0143
  max_NIDiasABP      0.2995               0.1149
      max_NIMAP      0.3394               0.1081
   max_NISysABP      0.5512               0.1087
         max_Na      0.3550               0.0820
      max_PaCO2      0.3362               0.1026
       max_PaO2      0.4671               0.3232
  max_Platelets      0.2016               0.1113
   max_RespRate      0.1719               0.0000
       max_SaO2      0.9419               0.0092
     max_SysABP      0.5436               0.0644
       max_Temp      0.3609               0.1538
  max_TroponinI      0.1938               0.0000
  max_TroponinT      0.0579               0.0000
      max_Urine      0.0781               0.0562
        max_WBC      0.0018               0.0006
         max_pH      0.0048               0.0015



