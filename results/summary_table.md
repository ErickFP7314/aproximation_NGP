| method                                           |   alpha_NGP |   delta_NGP |    std_alpha |   std_delta |   error_vs_iau_deg |
|:-------------------------------------------------|------------:|------------:|-------------:|------------:|-------------------:|
| IAU reference                                    |     12.85   |     27.13   | nan          |  nan        |            0       |
| great_circle_pole (flagship, sin distancia)      |     12.9463 |     26.4924 | nan          |  nan        |            1.43775 |
| aprox_ar_svd (SVD 2D, sin prior)                 |     12.9463 |     26.4924 |   0.00297275 |    0.041011 |            1.43775 |
| aprox_dec2 (ventana AR)                          |     12.9605 |     28.3286 |   0.291369   |    6.85157  |            1.89435 |
| ngp_3d_ransac (rt=2.0, con distancia 1/paralaje) |     13.2965 |     28.1176 | nan          |  nan        |            6.01541 |
| aprox_dec1 (top-n |dec|)                         |     12.4294 |     22.6581 |   5.54045    |    3.51758  |            7.26074 |