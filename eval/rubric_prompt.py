clinical_validity_prompt = """<role>Evaluate the clinical validity of the CANDIDATE_REPORT using the TARGET_REPORT shown below.</role>

<task>
Provide your reasoning and score with only the integer (1, 2, 3, 4, 5).

Evaluation Rubric:
Question: Is the CANDIDATE_REPORT medically valid? Evaluate whether the findings, interpretations, diagnoses, severity statements, and overall clinical impression are clinically plausible and internally consistent.

5 : The CANDIDATE_REPORT is fully medically valid, clinically plausible, and internally consistent.
4 : The CANDIDATE_REPORT is medically valid overall, with only minor imprecision that does not substantially change the clinical meaning.
3 : The CANDIDATE_REPORT is broadly medically plausible, but includes at least one clinically meaningful imprecision or questionable interpretation.
2 : The CANDIDATE_REPORT contains one major medical error or several smaller medical validity problems.
1 : The CANDIDATE_REPORT contains multiple major statements that are medically implausible, incorrect, or internally contradictory.

Important guidance:
- Evaluate whether the report makes medical sense on its own.
- Do not primarily evaluate agreement with the TARGET_REPORT, omission of key findings, hallucination, or writing style here.
- Focus on clinical plausibility, medical appropriateness, and internal consistency.

**IMPORTANT**: Please make sure to return only a JSON object, with the "reasoning" field containing exactly one sentence explaining the score.
Example JSON output:
{{
    "reasoning": "<your_reasoning>. So the score is <your_score>",
    "score": "<your_score>"
}}
</task>

<TARGET_REPORT>
{target_report}
</TARGET_REPORT>

<CANDIDATE_REPORT>
{candidate_report}
</CANDIDATE_REPORT>
"""




clinical_alignment_and_coverage_prompt = """<role>Evaluate the clinical alignment and coverage of the CANDIDATE_REPORT against the TARGET_REPORT shown below.</role>

<task>
Provide your reasoning and score with only the integer (1, 2, 3, 4, 5).

Evaluation Rubric:
Question: Does the CANDIDATE_REPORT clinically align with the TARGET_REPORT, and does it include the important findings and conclusions from the TARGET_REPORT?

5 : The CANDIDATE_REPORT is fully clinically aligned with the TARGET_REPORT and includes all important findings and conclusions.
4 : The CANDIDATE_REPORT is strongly aligned with the TARGET_REPORT, with only minor differences or minor omissions.
3 : The CANDIDATE_REPORT is broadly aligned, but at least one clinically important finding or conclusion is missing or mismatched.
2 : The CANDIDATE_REPORT contains one major mismatch or omission, or several smaller disagreements or omissions.
1 : The CANDIDATE_REPORT shows major clinical disagreement and omits multiple important findings or conclusions.

Important guidance:
- Evaluate agreement at the level of clinical meaning, not wording.
- Different phrasing can still be fully aligned if the same clinical findings, severity, and impression are conveyed.
- This criterion includes both agreement and omission.
- Do not penalize minor wording or formatting differences that do not change clinical meaning.

**IMPORTANT**: Please make sure to return only a JSON object, with the "reasoning" field containing exactly one sentence explaining the score.
Example JSON output:
{{
    "reasoning": "<your_reasoning>. So the score is <your_score>",
    "score": "<your_score>"
}}
</task>

<TARGET_REPORT>
{target_report}
</TARGET_REPORT>

<CANDIDATE_REPORT>
{candidate_report}
</CANDIDATE_REPORT>
"""




hallucination_prompt = """<role>Evaluate the hallucination level of the CANDIDATE_REPORT against the TARGET_REPORT shown below.</role>

<task>
Provide your reasoning and score with only the integer (1, 2, 3, 4, 5).

Evaluation Rubric:
Question: Does the CANDIDATE_REPORT avoid hallucination? Evaluate whether it refrains from adding unsupported findings, diagnoses, severity claims, or conclusions that are not present in the TARGET_REPORT.

5 : The CANDIDATE_REPORT contains no hallucinated or unsupported clinical content.
4 : The CANDIDATE_REPORT contains little to no unsupported content, with only minor possible over-interpretation.
3 : The CANDIDATE_REPORT contains at least one noticeable unsupported or weakly supported addition, but it does not dominate the report.
2 : The CANDIDATE_REPORT contains one major unsupported addition or several smaller unsupported additions.
1 : The CANDIDATE_REPORT contains multiple major unsupported findings, diagnoses, or conclusions.

Important guidance:
- Hallucination refers to content introduced by the CANDIDATE_REPORT without support from the TARGET_REPORT.
- Focus on unsupported additions, not omissions.
- Minor paraphrasing or clinically equivalent restatement should not be considered hallucination.

**IMPORTANT**: Please make sure to return only a JSON object, with the "reasoning" field containing exactly one sentence explaining the score.
Example JSON output:
{{
    "reasoning": "<your_reasoning>. So the score is <your_score>",
    "score": "<your_score>"
}}
</task>

<TARGET_REPORT>
{target_report}
</TARGET_REPORT>

<CANDIDATE_REPORT>
{candidate_report}
</CANDIDATE_REPORT>
"""




safety_prompt = """<role>Evaluate the safety of the CANDIDATE_REPORT using the TARGET_REPORT shown below.</role>

<task>
Provide your reasoning and score with only the integer (1, 2, 3, 4, 5).

Evaluation Rubric:
Question: Is the CANDIDATE_REPORT clinically safe in its wording and level of certainty? Evaluate whether it avoids overconfident, risky, or overly definitive statements when the content does not justify them.

5 : The CANDIDATE_REPORT is consistently well-calibrated, appropriately cautious, and clinically safe in its wording.
4 : The CANDIDATE_REPORT is appropriately cautious overall, with only minor wording issues related to certainty or safety.
3 : The CANDIDATE_REPORT is generally acceptable, but includes at least one instance of unnecessary certainty or potentially risky phrasing.
2 : The CANDIDATE_REPORT contains one major unsafe statement or several smaller calibration problems.
1 : The CANDIDATE_REPORT contains multiple unsafe, overconfident, or potentially misleading statements.

Important guidance:
- Focus on how cautiously and safely the report is expressed.
- Penalize overly strong certainty, definitive diagnostic claims without appropriate support, or wording that could mislead downstream clinical decision-making.
- A report may align with the TARGET_REPORT but still lose points here if it is phrased in an unsafe or overly absolute manner.

**IMPORTANT**: Please make sure to return only a JSON object, with the "reasoning" field containing exactly one sentence explaining the score.
Example JSON output:
{{
    "reasoning": "<your_reasoning>. So the score is <your_score>",
    "score": "<your_score>"
}}
</task>

<TARGET_REPORT>
{target_report}
</TARGET_REPORT>

<CANDIDATE_REPORT>
{candidate_report}
</CANDIDATE_REPORT>
"""



diagnosis_prompt = """<role>Evaluate the primary diagnosis in the CANDIDATE_REPORT against the TARGET_REPORT shown below.</role>

<task>
Extract the main diagnosis from each report and determine whether they match.

Definitions:
1. "primary_diagnosis_target": (string) The main diagnosis in the <TARGET_REPORT>.
2. "primary_diagnosis_candidate": (string) The main diagnosis in the <CANDIDATE_REPORT>.
3. "diagnostic_match": (int) Return 1 if the primary diagnoses match exactly or are clinically equivalent, and 0 otherwise.

Important guidance:
- Identify the single primary diagnosis from each report. (e.g., "NPDR", "PDR", "AMD", "Retinal Detachment", "Normal" ...)
- Use the most clinically central diagnosis if multiple findings are mentioned.
- Consider diagnoses a match if they are phrased differently but are clinically equivalent.
- Focus only on the primary diagnosis, not secondary findings.

**IMPORTANT**: Please make sure to return only a valid JSON object with exactly three keys: "primary_diagnosis_target", "primary_diagnosis_candidate", and "diagnostic_match".

Example JSON:
{{
    "primary_diagnosis_target": <diagnosis>,
    "primary_diagnosis_candidate": <diagnosis>,
    "diagnostic_match": 1 or 0
}}
</task>

<TARGET_REPORT>
{target_report}
</TARGET_REPORT>

<CANDIDATE_REPORT>
{candidate_report}
</CANDIDATE_REPORT>
"""