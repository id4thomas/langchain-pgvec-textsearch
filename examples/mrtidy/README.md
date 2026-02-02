# MrTidy Example

Multilingual retrieval evaluation using the MrTidy dataset

## Perf

### Korean
- **Korean**: 1.5M documents, 421 test queries
```
--- Results @ k=5 ---
                          Recall@5  Precision@5  nDCG@5    MRR    MAP
Config                                                               
dense                       0.4877       0.1074  0.3965 0.3971 0.3793
sparse (BM25 korean)        0.2118       0.0466  0.1834 0.1890 0.1793
hybrid (RRF k=60)           0.3654       0.0822  0.3011 0.3145 0.2987
hybrid (RRF k=60, d=0.7)    0.4375       0.0969  0.3403 0.3435 0.3278
hybrid (RRF k=30, d=0.7)    0.4861       0.1083  0.3735 0.3660 0.3500
hybrid (RRF k=20)           0.4125       0.0917  0.3266 0.3333 0.3166

--- Results @ k=10 ---
                          Recall@10  Precision@10  nDCG@10    MRR    MAP
Config                                                                  
dense                        0.5760        0.0648   0.4268 0.3971 0.3793
sparse (BM25 korean)         0.2534        0.0280   0.1974 0.1890 0.1793
hybrid (RRF k=60)            0.4850        0.0544   0.3408 0.3145 0.2987
hybrid (RRF k=60, d=0.7)     0.5503        0.0620   0.3789 0.3435 0.3278
hybrid (RRF k=30, d=0.7)     0.5661        0.0637   0.4006 0.3660 0.3500
hybrid (RRF k=20)            0.5321        0.0594   0.3667 0.3333 0.3166

--- Results @ k=20 ---
                          Recall@20  Precision@20  nDCG@20    MRR    MAP
Config                                                                  
dense                        0.6667        0.0378   0.4508 0.3971 0.3793
sparse (BM25 korean)         0.2842        0.0157   0.2061 0.1890 0.1793
hybrid (RRF k=60)            0.5903        0.0336   0.3687 0.3145 0.2987
hybrid (RRF k=60, d=0.7)     0.6623        0.0375   0.4086 0.3435 0.3278
hybrid (RRF k=30, d=0.7)     0.6706        0.0380   0.4287 0.3660 0.3500
hybrid (RRF k=20)            0.6271        0.0355   0.3920 0.3333 0.3166

======================================================================
Summary Table
======================================================================
                            R@5  nDCG@5   R@10  nDCG@10   R@20  nDCG@20    MRR    MAP
Config                                                                               
dense                    0.4877  0.3965 0.5760   0.4268 0.6667   0.4508 0.3971 0.3793
sparse (BM25 korean)     0.2118  0.1834 0.2534   0.1974 0.2842   0.2061 0.1890 0.1793
hybrid (RRF k=60)        0.3654  0.3011 0.4850   0.3408 0.5903   0.3687 0.3145 0.2987
hybrid (RRF k=60, d=0.7) 0.4375  0.3403 0.5503   0.3789 0.6623   0.4086 0.3435 0.3278
hybrid (RRF k=30, d=0.7) 0.4861  0.3735 0.5661   0.4006 0.6706   0.4287 0.3660 0.3500
hybrid (RRF k=20)        0.4125  0.3266 0.5321   0.3667 0.6271   0.3920 0.3333 0.3166
```

### English
- **English**: 32.9M documents, 744 test queries

```
```
