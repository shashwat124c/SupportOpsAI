SupportOpsAI — Final QLoRA Evaluation
=======================================

Model:
Qwen/Qwen2.5-1.5B-Instruct

Method:
4-bit QLoRA

Test set:
2,375 untouched test examples

Results:
Queue Accuracy:       36.97%
Queue Macro-F1:       19.41%
Priority Accuracy:    47.54%
Priority Macro-F1:    35.27%
Joint Accuracy:       21.05%
JSON Validity:       100.00%

Baseline → QLoRA:
Queue Accuracy:       31.66% → 36.97%
Queue Macro-F1:       20.40% → 19.41%
Priority Accuracy:    37.81% → 47.54%
Priority Macro-F1:    23.75% → 35.27%
Joint Accuracy:       15.24% → 21.05%
JSON Validity:        95.87% → 100.00%

Main observation:
The QLoRA model substantially improved priority classification and
joint accuracy. Queue classification remained challenging, with
a strong prediction bias toward the Technical Support class.
