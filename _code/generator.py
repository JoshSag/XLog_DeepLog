import numpy as np
import math

def make_vocabulary(vocabulary_size):
    digits = int(math.floor(1+math.log(vocabulary_size, 10)))
    vocabulary = ["s{}".format(str(i).zfill(digits)) for i in range(vocabulary_size)]
    return vocabulary

def generate_pattern(vocabulary, min_pattern_size, max_pattern_size):
    return tuple(np.random.choice(vocabulary,np.random.randint(min_pattern_size,max_pattern_size+1)))

def generate_patterns(num_patterns, vocabulary, min_pattern_size, max_pattern_size):
    patterns = [generate_pattern(vocabulary, min_pattern_size, max_pattern_size) for _ in range(num_patterns)]
    return patterns

def generate_text(patterns, text_size, anomaly_ratio, vocabulary, patterns_p = None):
    p_i = np.random.choice(range(len(patterns)), size=text_size, p=patterns_p)
    
    text = list()
    marks = list()
    for j,i in enumerate(p_i):
        p = patterns[i]
        text.extend(p)
        marks.extend(list(range(1,len(p)+1)))

        if j < len(p_i) -1: # no injection after the last pattern
            inject_anomaly = np.random.rand() < anomaly_ratio
            if inject_anomaly:
                anomaly_length=1
                anomaly = np.random.choice(vocabulary,anomaly_length)
                text.extend(anomaly)
                marks.extend([0]*len(anomaly))
    
    return text, marks

def generate_tests(patterns, vocabulary, n, text_size, anomaly_ratio, patterns_p=None):
    tests = dict()
    for i in range(n):
        text_test , marks_test = generate_text(patterns, text_size=text_size , anomaly_ratio=anomaly_ratio, vocabulary=vocabulary, patterns_p=patterns_p)
        anomaly = 0 in marks_test
        tests[i] = (text_test, marks_test, anomaly)
    return tests