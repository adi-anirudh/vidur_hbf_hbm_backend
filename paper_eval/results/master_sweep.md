
============================================================================================
A. MIN GPUs to hit TBT<=100ms
============================================================================================

--- ctx=8K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   13      9     49     58
   B2                  -    -    8     11     80     91
   B4                  1    8    9     42     95     95
   B5                  1    8    9     42     95     95
   Ours.HBF            1    6    7     42     98     98
   Ours.HBF+p          1    5    6     42     95     95
   Ours.HBM/HBF        1    6    7     42     98     98
   Ours.HBM/HBF+p      1    5    6     42     95     95

--- ctx=8K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   13     12     50     62
   B2                  -    -    8     15     82     97
   B4                  1    9   10     78     86     86
   B5                  2    9   11     42     86     86
   Ours.HBF            1    7    8     78     91     91
   Ours.HBF+p          1    6    7     78     89     89
   Ours.HBM/HBF        2    7    9     42     91     91
   Ours.HBM/HBF+p      2    6    8     42     89     89

--- ctx=8K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   15     16     45     61
   B2                  -    -    9     22     75     97
   B4                  2    9   11     78     91     91
   B5                  3    9   12     54     91     91
   Ours.HBF            2    7    9     78     96     96
   Ours.HBF+p          2    6    8     78     98     98
   Ours.HBM/HBF        3    7   10     54     96     96
   Ours.HBM/HBF+p      3    6    9     54     98     98

--- ctx=8K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   17     24     41     65
   B2                  -    -   11     33     64     97
   B4                  4    9   13     78     96     96
   B5                  6    9   15     54     96     96
   Ours.HBF            4    8   12     78     89     89
   Ours.HBF+p          4    7   11     78     93     93
   Ours.HBM/HBF        6    8   14     54     89     89
   Ours.HBM/HBF+p      6    7   13     54     93     93

--- ctx=16K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   13     12     49     61
   B2                  -    -    8     15     80     95
   B4                  1    8    9     78     95     95
   B5                  2    8   10     42     95     95
   Ours.HBF            1    6    7     78     98     98
   Ours.HBF+p          1    5    6     78     95     95
   Ours.HBM/HBF        2    6    8     42     98     98
   Ours.HBM/HBF+p      2    5    7     42     95     95

--- ctx=16K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   15     16     44     60
   B2                  -    -    9     23     73     96
   B4                  2    9   11     78     86     86
   B5                  3    9   12     55     86     86
   Ours.HBF            2    7    9     78     91     91
   Ours.HBF+p          2    6    8     78     89     89
   Ours.HBM/HBF        3    7   10     55     91     91
   Ours.HBM/HBF+p      3    6    9     55     89     89

--- ctx=16K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   17     24     40     63
   B2                  -    -   11     33     61     94
   B4                  4    9   13     78     91     91
   B5                  6    9   15     55     91     91
   Ours.HBF            4    7   11     78     96     96
   Ours.HBF+p          4    6   10     78     98     98
   Ours.HBM/HBF        6    7   13     55     96     96
   Ours.HBM/HBF+p      6    6   12     55     98     98

--- ctx=16K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   22     33     32     65
   B2                  -    -   14     48     50     98
   B4                  7    9   16     89     96     96
   B5                 11    9   20     59     96     96
   Ours.HBF            7    8   15     89     89     89
   Ours.HBF+p          7    7   14     89     93     93
   Ours.HBM/HBF       11    8   19     59     89     89
   Ours.HBM/HBF+p     11    7   18     59     93     93

--- ctx=32K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   15     16     43     59
   B2                  -    -    9     23     71     94
   B4                  2    8   10     78     95     95
   B5                  3    8   11     55     95     95
   Ours.HBF            2    6    8     78     98     98
   Ours.HBF+p          2    5    7     78     95     95
   Ours.HBM/HBF        3    6    9     55     98     98
   Ours.HBM/HBF+p      3    5    8     55     95     95

--- ctx=32K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   18     23     36     60
   B2                  -    -   11     33     60     93
   B4                  4    9   13     78     86     86
   B5                  6    9   15     55     86     86
   Ours.HBF            4    7   11     78     91     91
   Ours.HBF+p          4    6   10     78     89     89
   Ours.HBM/HBF        6    7   13     55     91     91
   Ours.HBM/HBF+p      6    6   12     55     89     89

--- ctx=32K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   23     32     29     61
   B2                  -    -   14     48     48     96
   B4                  7    9   16     90     91     91
   B5                 11    9   20     59     91     91
   Ours.HBF            7    7   14     90     96     96
   Ours.HBF+p          7    6   13     90     98     98
   Ours.HBM/HBF       11    7   18     59     96     96
   Ours.HBM/HBF+p     11    6   17     59     98     98

--- ctx=32K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   34     41     21     62
   B2                  -    -   20     65     35    100
   B4                 13    9   22     95     96     96
   B5                 21    9   30     61     96     96
   Ours.HBF           13    8   21     95     89     95
   Ours.HBF+p         13    7   20     95     93     95
   Ours.HBM/HBF       21    8   29     61     89     89
   Ours.HBM/HBF+p     21    7   28     61     93     93

--- ctx=64K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   19     22     34     56
   B2                  -    -   10     36     64     99
   B4                  4    8   12     78     95     95
   B5                  6    8   14     56     95     95
   Ours.HBF            4    6   10     78     98     98
   Ours.HBF+p          4    5    9     78     95     95
   Ours.HBM/HBF        6    6   12     56     98     98
   Ours.HBM/HBF+p      6    5   11     56     95     95

--- ctx=64K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   24     31     27     58
   B2                  -    -   14     49     47     96
   B4                  7    9   16     90     86     90
   B5                 11    9   20     60     86     86
   Ours.HBF            7    7   14     90     91     91
   Ours.HBF+p          7    6   13     90     89     90
   Ours.HBM/HBF       11    7   18     60     91     91
   Ours.HBM/HBF+p     11    6   17     60     89     89

--- ctx=64K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   35     40     19     59
   B2                  -    -   20     65     34     98
   B4                 13    9   22     96     91     96
   B5                 22    9   31     60     91     91
   Ours.HBF           13    7   20     96     96     96
   Ours.HBF+p         13    6   19     96     98     98
   Ours.HBM/HBF       22    7   29     60     96     96
   Ours.HBM/HBF+p     22    6   28     60     98     98

--- ctx=64K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   54     49     13     62
   B2                  -    -   34     76     21     97
   B4                 25    9   34     99     96     99
   B5                 43    9   52     60     96     96
   Ours.HBF           25    8   33     99     89     99
   Ours.HBF+p         25    7   32     99     93     99
   Ours.HBM/HBF       43    8   51     60     89     89
   Ours.HBM/HBF+p     43    7   50     60     93     93

--- ctx=128K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   26     29     25     53
   B2                  -    -   15     47     43     89
   B4                  7    8   15     92     95     95
   B5                 11    8   19     60     95     95
   Ours.HBF            7    6   13     92     98     98
   Ours.HBF+p          7    5   12     92     95     95
   Ours.HBM/HBF       11    6   17     60     98     98
   Ours.HBM/HBF+p     11    5   16     60     95     95

--- ctx=128K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   37     38     18     56
   B2                  -    -   20     65     33     98
   B4                 13    9   22     96     86     96
   B5                 22    9   31     60     86     86
   Ours.HBF           13    7   20     96     91     96
   Ours.HBF+p         13    6   19     96     89     96
   Ours.HBM/HBF       22    7   29     60     91     91
   Ours.HBM/HBF+p     22    6   28     60     89     89

--- ctx=128K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   57     47     12     59
   B2                  -    -   32     78     21     99
   B4                 26    9   35     96     91     96
   B5                 43    9   52     60     91     91
   Ours.HBF           26    7   33     96     96     96
   Ours.HBF+p         26    6   32     96     98     98
   Ours.HBM/HBF       43    7   50     60     96     96
   Ours.HBM/HBF+p     43    6   49     60     98     98

--- ctx=128K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -  103     51      7     58
   B2                  -    -   57     87     12    100
   B4                 52    9   61     96     96     96
   B5                 86    9   95     60     96     96
   Ours.HBF           52    8   60     96     89     96
   Ours.HBF+p         52    7   59     96     93     96
   Ours.HBM/HBF       86    8   94     60     89     89
   Ours.HBM/HBF+p     86    7   93     60     93     93

============================================================================================
B. BEST latency at FIXED per-S budget {128: 16, 256: 28, 512: 48, 1024: 90}
============================================================================================

--- ctx=8K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   16      9     40     48
   B2                  -    -   16      9     40     48
   B4                  1   15   16     42     57     57
   B5                  1   15   16     42     57     57
   Ours.HBF            1   15   16     42     41     42
   Ours.HBF+p          2   14   16     24     36     36
   Ours.HBM/HBF        1   15   16     42     41     42
   Ours.HBM/HBF+p      2   14   16     24     36     36

--- ctx=8K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   26      9     25     34
   B2                  -    -   26      9     25     34
   B4                  3   25   28     31     42     42
   B5                  3   25   28     31     42     42
   Ours.HBF            4   24   28     24     28     28
   Ours.HBF+p          4   24   28     24     24     24
   Ours.HBM/HBF        4   24   28     24     28     28
   Ours.HBM/HBF+p      4   24   28     24     24     24

--- ctx=8K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   47      9     14     24
   B2                  -    -   47      9     14     24
   B4                  6   42   48     31     31     31
   B5                  6   42   48     31     31     31
   Ours.HBF           11   37   48     20     20     20
   Ours.HBF+p         12   36   48     18     18     18
   Ours.HBM/HBF       11   37   48     20     20     20
   Ours.HBM/HBF+p     12   36   48     18     18     18

--- ctx=8K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   86     10      8     18
   B2                  -    -   86     10      8     18
   B4                 19   71   90     22     25     25
   B5                 19   71   90     22     25     25
   Ours.HBF           34   56   90     15     15     15
   Ours.HBF+p         36   54   90     15     14     15
   Ours.HBM/HBF       34   56   90     15     15     15
   Ours.HBM/HBF+p     36   54   90     15     14     15

--- ctx=16K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   16     11     40     51
   B2                  -    -   16     11     40     51
   B4                  2   14   16     42     61     61
   B5                  2   14   16     42     61     61
   Ours.HBF            2   14   16     42     43     43
   Ours.HBF+p          3   13   16     31     38     38
   Ours.HBM/HBF        2   14   16     42     43     43
   Ours.HBM/HBF+p      3   13   16     31     38     38

--- ctx=16K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   26     12     25     37
   B2                  -    -   26     12     25     37
   B4                  4   24   28     42     44     44
   B5                  4   24   28     42     44     44
   Ours.HBF            6   22   28     31     31     31
   Ours.HBF+p          7   21   28     27     27     27
   Ours.HBM/HBF        6   22   28     31     31     31
   Ours.HBM/HBF+p      7   21   28     27     27     27

--- ctx=16K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   47     13     14     27
   B2                  -    -   47     13     14     27
   B4                 12   36   48     31     33     33
   B5                 12   36   48     31     33     33
   Ours.HBF           17   31   48     24     24     24
   Ours.HBF+p         19   29   48     22     22     22
   Ours.HBM/HBF       17   31   48     24     24     24
   Ours.HBM/HBF+p     19   29   48     22     22     22

--- ctx=16K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   86     13      8     21
   B2                  -    -   86     13      8     21
   B4                 29   61   90     27     27     27
   B5                 29   61   90     27     27     27
   Ours.HBF           47   43   90     19     19     19
   Ours.HBF+p         49   41   90     18     18     18
   Ours.HBM/HBF       47   43   90     19     19     19
   Ours.HBM/HBF+p     49   41   90     18     18     18

--- ctx=32K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   16     15     40     55
   B2                  -    -   16     15     40     55
   B4                  3   13   16     55     62     62
   B5                  3   13   16     55     62     62
   Ours.HBF            4   12   16     42     50     50
   Ours.HBF+p          4   12   16     42     41     42
   Ours.HBM/HBF        4   12   16     42     50     50
   Ours.HBM/HBF+p      4   12   16     42     41     42

--- ctx=32K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   26     18     25     43
   B2                  -    -   26     18     25     43
   B4                  8   20   28     42     49     49
   B5                  8   20   28     42     49     49
   Ours.HBF           10   18   28     36     37     37
   Ours.HBF+p         11   17   28     33     33     33
   Ours.HBM/HBF       10   18   28     36     37     37
   Ours.HBM/HBF+p     11   17   28     33     33     33

--- ctx=32K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   47     19     14     33
   B2                  -    -   47     19     14     33
   B4                 18   30   48     39     40     40
   B5                 18   30   48     39     40     40
   Ours.HBF           25   23   48     30     31     31
   Ours.HBF+p         26   22   48     29     29     29
   Ours.HBM/HBF       25   23   48     30     31     31
   Ours.HBM/HBF+p     26   22   48     29     29     29

--- ctx=32K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   86     20      8     28
   B2                  -    -   86     20      8     28
   B4                 47   43   90     31     32     32
   B5                 47   43   90     31     32     32
   Ours.HBF           61   29   90     25     26     26
   Ours.HBF+p         61   29   90     25     24     25
   Ours.HBM/HBF       61   29   90     25     26     26
   Ours.HBM/HBF+p     61   29   90     25     24     25

--- ctx=64K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  ×                               
   B2                  -    -   16     24     40     64
   B4                  5   11   16     65     71     71
   B5                  6   10   16     56     77     77
   Ours.HBF            6   10   16     56     60     60
   Ours.HBF+p          7    9   16     49     54     54
   Ours.HBM/HBF        6   10   16     56     60     60
   Ours.HBM/HBF+p      7    9   16     49     54     54

--- ctx=64K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   26     29     25     54
   B2                  -    -   26     29     25     54
   B4                 12   16   28     56     59     59
   B5                 12   16   28     56     59     59
   Ours.HBF           14   14   28     49     47     49
   Ours.HBF+p         16   12   28     42     46     46
   Ours.HBM/HBF       14   14   28     49     47     49
   Ours.HBM/HBF+p     16   12   28     42     46     46

--- ctx=64K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   47     31     14     45
   B2                  -    -   47     31     14     45
   B4                 26   22   48     51     49     51
   B5                 26   22   48     51     49     51
   Ours.HBF           32   16   48     42     43     43
   Ours.HBF+p         32   16   48     42     38     42
   Ours.HBM/HBF       32   16   48     42     43     43
   Ours.HBM/HBF+p     32   16   48     42     38     42

--- ctx=64K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  -    -   86     33      8     42
   B2                  -    -   86     33      8     42
   B4                 61   29   90     45     43     45
   B5                 61   29   90     45     43     45
   Ours.HBF           69   21   90     40     36     40
   Ours.HBF+p         69   21   90     40     33     40
   Ours.HBM/HBF       69   21   90     40     36     40
   Ours.HBM/HBF+p     69   21   90     40     33     40

--- ctx=128K S=128  (E_uniq=227, Ekeep@.98=182) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  ×                               
   B2                  -    -   16     42     40     82
   B4                  7    9   16     92     79     92
   B5                 11    5   16     60    137    137
   Ours.HBF            8    8   16     78     74     78
   Ours.HBF+p          9    7   16     74     68     74
   Ours.HBM/HBF       11    5   16     60    117    117
   Ours.HBM/HBF+p     11    5   16     60     95     95

--- ctx=128K S=256  (E_uniq=242, Ekeep@.98=202) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  ×                               
   B2                  -    -   26     51     25     77
   B4                 16   12   28     78     69     78
   B5                 22    6   28     60    129    129
   Ours.HBF           19    9   28     69     71     71
   Ours.HBF+p         20    8   28     65     67     67
   Ours.HBM/HBF       22    6   28     60    105    105
   Ours.HBM/HBF+p     22    6   28     60     89     89

--- ctx=128K S=512  (E_uniq=250, Ekeep@.98=218) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  ×                               
   B2                  -    -   47     56     14     70
   B4                 35   13   48     74     69     74
   B5                 43    5   48     60    156    156
   Ours.HBF           37   11   48     69     62     69
   Ours.HBF+p         37   11   48     69     55     69
   Ours.HBM/HBF       43    5   48     60    133    133
   Ours.HBM/HBF+p     43    5   48     60    117    117

--- ctx=128K S=1024  (E_uniq=254, Ekeep@.98=231) ---
   system              A    M    G   Tatt   Tmoe   step
   B1                  ×                               
   B2                  -    -   86     60      8     69
   B4                 74   16   90     69     64     69
   B5                 86    4   90     60    209    209
   Ours.HBF           79   11   90     65     65     65
   Ours.HBF+p         79   11   90     65     60     65
   Ours.HBM/HBF       86    4   90     60    175    175
   Ours.HBM/HBF+p     86    4   90     60    160    160