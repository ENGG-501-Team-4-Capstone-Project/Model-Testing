# Benchmark

Calls: 60 | Images: 5 | Runs per image per model: 3

## Overall by model
```
                                  full_image_accuracy json_valid_rate  errors  avg_latency_s  p95_latency_s  avg_cost_usd  total_cost_usd  avg_prompt_tokens  avg_completion_tokens field_accuracy  cost_per_correct_field_usd consistency
model                                                                                                                                                                                                                                     
nvidia/nemotron-3.5-super-vl:free               40.0%          100.0%       0        1.96888        3.14409       0.00000         0.00000             1030.2              117.00000          86.7%                     0.00000       80.0%
deepseek/deepseek-v4.1-flash-1017               60.0%          100.0%       0        2.30160        7.49609       0.00009         0.00131              711.8               75.20000          88.9%                     0.00001       80.0%
anthropic/claude-haiku-5.5                      46.7%          100.0%       0        1.76533        2.35493       0.00019         0.00287             1356.2              115.40000          90.4%                     0.00002       40.0%
openai/gpt-6-luna                               40.0%          100.0%       0        2.35210        3.26341       0.00008         0.00126             1169.0               73.93333          82.2%                     0.00001       60.0%
```

## Field accuracy (%) by image
```
model  anthropic/claude-haiku-5.5  deepseek/deepseek-v4.1-flash-1017  nvidia/nemotron-3.5-super-vl:free  openai/gpt-6-luna
image                                                                                                                     
1.jpg                       100.0                              100.0                              100.0               75.0
2.jpg                       100.0                              100.0                               90.9              100.0
3.jpg                        77.8                               66.7                               66.7               66.7
4.jpg                        81.0                              100.0                              100.0              100.0
5.jpg                        90.0                               80.0                               80.0               70.0
```

## Hardest fields (% correct)
```
image  field        
3.jpg  serial_number     0.0
       brand             0.0
5.jpg  serial_number     0.0
3.jpg  type             25.0
5.jpg  date             58.3
       length           66.7
2.jpg  manufacturer     75.0
1.jpg  manufacturer     75.0
       serial_number    75.0
4.jpg  wll_left_t       83.3
```

## Mismatches (first 40)
```
                            model image  run         field    expected        predicted  correct
nvidia/nemotron-3.5-super-vl:free 2.jpg    1  manufacturer     UNIROPE UNIROPE SLINGMAX    False
nvidia/nemotron-3.5-super-vl:free 2.jpg    2  manufacturer     UNIROPE UNIROPE SLINGMAX    False
nvidia/nemotron-3.5-super-vl:free 2.jpg    3  manufacturer     UNIROPE UNIROPE SLINGMAX    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    1         brand   WEBMASTER         WEMASTER    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    1          type  3 EE1-804D       3 EE1-6040    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    1 serial_number    853131-1         S.N 5131    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    2         brand   WEBMASTER      WESTMARSTER    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    2          type  3 EE1-804D       3 EE1-6040    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    2 serial_number    853131-1         S.N 5131    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    3         brand   WEBMASTER        WEHMASTER    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    3          type  3 EE1-804D       3 EE1-6040    False
nvidia/nemotron-3.5-super-vl:free 3.jpg    3 serial_number    853131-1         S.N 5131    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    1          date       08-20            03-20    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    1 serial_number 30P30619094    38P3861-9-034    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    2          date       08-20            03-20    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    2 serial_number 30P30619094    38P3861-9-034    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    3          date       08-20            03-20    False
nvidia/nemotron-3.5-super-vl:free 5.jpg    3 serial_number 30P30619094    38P3861-9-034    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    1         brand   WEBMASTER      WEINBRENNER    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    1          type  3 EE1-804D       3 EE1-8040    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    1 serial_number    853131-1             35-1    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    2         brand   WEBMASTER      WEINBRENNER    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    2          type  3 EE1-804D       3 EE1-8040    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    2 serial_number    853131-1             35-1    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    3         brand   WEBMASTER      WEINBRENNER    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    3          type  3 EE1-804D       3 EE1-8040    False
deepseek/deepseek-v4.1-flash-1017 3.jpg    3 serial_number    853131-1             35-1    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    1        length           8             None    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    1 serial_number 30P30619094    30P30661S1634    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    2        length           8             None    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    2 serial_number 30P30619094  30P30661-3-1-64    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    3        length           8             None    False
deepseek/deepseek-v4.1-flash-1017 5.jpg    3 serial_number 30P30619094  30P30661-3-1-64    False
       anthropic/claude-haiku-5.5 3.jpg    1         brand   WEBMASTER       Werimaster    False
       anthropic/claude-haiku-5.5 3.jpg    1 serial_number    853131-1           5331-1    False
       anthropic/claude-haiku-5.5 3.jpg    2         brand   WEBMASTER       Werma-Ster    False
       anthropic/claude-haiku-5.5 3.jpg    2 serial_number    853131-1           5301-1    False
       anthropic/claude-haiku-5.5 3.jpg    3         brand   WEBMASTER        Wesmaster    False
       anthropic/claude-haiku-5.5 3.jpg    3 serial_number    853131-1           5303-1    False
       anthropic/claude-haiku-5.5 4.jpg    1    wll_left_t         3.2                3    False
```